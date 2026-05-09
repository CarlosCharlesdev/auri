from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
import whisper
import edge_tts
import httpx
import base64
import os
import tempfile
import subprocess
import random
from datetime import datetime
from contextlib import asynccontextmanager

import asyncpg

load_dotenv()

def normalizar_resposta(texto: str) -> str:
    """Normaliza texto transcrito: remove acento, pontuacao, lowercase.
    Se contiver sim ou nao em qualquer lugar, retorna isso."""
    import unicodedata
    texto = texto.lower().strip()
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    texto = ''.join(c for c in texto if c.isalnum() or c == ' ')
    texto = texto.strip()
    palavras = texto.split()
    # Se contiver sim ou nao em qualquer posição, retorna isso
    if 'sim' in palavras:
        return 'sim'
    if 'nao' in palavras or 'n' in palavras:
        return 'nao'
    return palavras[0] if palavras else texto

def extrair_descricao_limpa_sync(mensagem: str) -> str:
    """Extrai descrição limpa via regex sem usar IA."""
    import re

    # 1. Conteúdo entre aspas — pega direto
    match = re.search(r'["“„](.+?)["”]', mensagem)
    if not match:
        match = re.search(r"'(.+?)'", mensagem)
    if match:
        return match.group(1).strip()

    # 2. Remove preâmbulos conhecidos
    preambulos = [
        r"^(gostaria de |quero |pode |por favor |pfv |pf )?(mudar|alterar|corrigir|trocar|atualizar)( a descri[çc][aã]o| o texto| isso| para| pra| s[oó]| apenas)?[:\s]*",
        r"^(errei[,\s]+|na verdade[,\s]+|[ée] na verdade[,\s]+|mentira[,\s]+)",
        r"^(o certo [ée][:\s]*|correto seria[:\s]*|deveria ser[:\s]*)",
        r"^(coloca[,\s]+|escreve[,\s]+|registra[,\s]+)",
        r"^(a descri[çc][aã]o [ée][:\s]*|descri[çc][aã]o[:\s]*)",
        r"^(muda[,\s]+|mudando[,\s]+)",
    ]
    texto = mensagem.strip()
    for p in preambulos:
        novo = re.sub(p, "", texto, flags=re.IGNORECASE).strip().lstrip(",:;- ").strip()
        if novo:
            texto = novo

    return texto if texto else mensagem

async def extrair_descricao_limpa(mensagem: str) -> str:
    """Extrai descrição limpa — regex, sem chamar IA."""
    return extrair_descricao_limpa_sync(mensagem)

def detectar_correcao(mensagem: str) -> str | None:
    """Detecta o que a pessoa quer corrigir. Retorna: 'tipo', 'local', 'foto' ou None."""
    import unicodedata
    texto = mensagem.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")

    # Só detecta "local" se a pessoa EXPLICITAMENTE quer mudar o endereço
    if any(p in texto for p in ["mudar o local", "mudar endereco", "alterar local",
                                  "alterar endereco", "corrigir local", "novo local",
                                  "errei o local", "errei a rua", "errei o endereco"]):
        return "local"
    if any(p in texto for p in ["foto", "imagem", "print", "fotografia"]):
        return "foto"
    # Qualquer outra coisa = quer corrigir a descricao/tipo
    return "tipo"

# ==============================
# 🔧 CONFIG
# ==============================
OLLAMA_URL    = os.getenv("OLLAMA_URL",       "http://localhost:11434/api/generate")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",     "llama3.2")
WHAPI_URL     = "https://gate.whapi.cloud"
WHAPI_TOKEN   = os.getenv("WHATSAPP_TOKEN",   "")
VOICE         = os.getenv("TTS_VOICE",        "pt-BR-FranciscaNeural")

DB_HOST       = os.getenv("DB_HOST",          "localhost")
DB_PORT       = int(os.getenv("DB_PORT",      "5432"))
DB_USER       = os.getenv("DB_USER",          "postgres")
DB_PASSWORD   = os.getenv("DB_PASSWORD",      "postgres")
DB_NAME       = os.getenv("DB_NAME",          "ia_vendas")

# ==============================
# 🌐 ESTADO GLOBAL
# ==============================
sessoes: dict = {}
db_pool = None
whisper_model = None

# ==============================
# 🚀 STARTUP / SHUTDOWN
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, whisper_model

    print("Carregando Whisper small (GPU)...")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Usando device: {device}")
    whisper_model = whisper.load_model("small", device=device)
    print("✅ Whisper pronto!")

    print("Conectando ao banco...")
    db_pool = await asyncpg.create_pool(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME
    )
    await init_db()
    print("✅ Banco pronto!")

    print("🚀 Bot de denúncias rodando!")
    yield

    await db_pool.close()

app = FastAPI(lifespan=lifespan)

# ==============================
# 🗄️ BANCO DE DADOS
# ==============================
async def init_db():
    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS denuncias (
                id               SERIAL PRIMARY KEY,
                protocolo        VARCHAR(20)  NOT NULL UNIQUE,
                numero_whatsapp  VARCHAR(50),
                modulo           VARCHAR(50),
                subcategoria     VARCHAR(100),
                descricao        TEXT,
                local            VARCHAR(255),
                data_ocorrencia  VARCHAR(100),
                anonimo          BOOLEAN      DEFAULT FALSE,
                foto_url         TEXT,
                urgente          BOOLEAN      DEFAULT FALSE,
                status           VARCHAR(20)  DEFAULT 'pendente',
                criado_em        TIMESTAMP    DEFAULT NOW()
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_denuncias_status    ON denuncias(status)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_denuncias_modulo    ON denuncias(modulo)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_denuncias_criado_em ON denuncias(criado_em)")

# ==============================
# 🚨 DETECÇÃO DE EMERGÊNCIA
# ==============================
EMERGENCIAS = {
    "violencia_domestica": {
        "palavras": [
            "apanhando", "me bate", "me bateu", "marido bate", "companheiro bate",
            "namorado bate", "esposo bate", "violencia domestica", "violencia domestica",
            "me agride", "me agrediu", "me socorre", "socorro mulher", "lei maria da penha",
            "me ameacando", "me ameacando", "ameaca de morte", "ameaca de morte",
            "feminicidio", "feminicidio", "mulher apanhando", "ele me bate",
            "ela me bate", "meu marido me", "meu namorado me", "meu companheiro me",
            "ta me batendo", "esta me batendo", "to apanhando", "estou apanhando"
        ],
        "modulo": "Violencia Domestica",
        "subcategoria": "Violencia contra a mulher",
        "sigla": "VDM",
        "resposta": (
            "🚨 *ATENÇÃO — SITUAÇÃO DE RISCO DETECTADA*\n\n"
            "Se você estiver em perigo imediato, ligue agora:\n\n"
            "📞 *190* — Polícia Militar\n"
            "📞 *180* — Central de Atendimento à Mulher\n"
            "📞 *192* — SAMU\n\n"
            "Sua denúncia será registrada como *URGENTE* e encaminhada.\n\n"
            "📍 Me informe o endereço onde está ocorrendo."
        ),
        "falar": (
            "Atencao. Detectamos uma situacao de risco. "
            "Se voce estiver em perigo, ligue agora 190 para a policia "
            "ou 180 para a central de atendimento a mulher. "
            "Vou registrar sua denuncia como urgente. "
            "Me informe o endereco onde esta ocorrendo."
        )
    }
}

def detectar_emergencia(mensagem: str) -> dict | None:
    import unicodedata
    texto = mensagem.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    for tipo, dados in EMERGENCIAS.items():
        if any(p in texto for p in dados["palavras"]):
            return dados
    return None

SIGLAS_MODULO = {
    "Infraestrutura":        "INF",
    "Meio Ambiente":         "MEA",
    "Dano ao Patrimônio":    "DPA",
    "Saúde Pública":         "SAP",
    "Mobilidade / Trânsito": "MOB",
    "Serviços Públicos":     "SRP",
    "Outros":                "OUT",
}

def gerar_protocolo(modulo: str = "OUT", sigla_custom: str = None):
    import unicodedata
    sigla = sigla_custom or SIGLAS_MODULO.get(modulo)
    if not sigla:
        # Fallback: pega primeiras letras de cada palavra
        texto = unicodedata.normalize("NFD", modulo)
        texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
        sigla = "".join(p[0].upper() for p in texto.split() if p)[:3]
    ano = datetime.now().year
    num = str(random.randint(100, 999))
    seq = str(random.randint(10, 99))
    return f"{sigla}-{num}-{seq}"

async def registrar_denuncia(dados: dict) -> str:
    sigla_custom = dados.get("sigla")
    protocolo = gerar_protocolo(dados.get("modulo", "Outros"), sigla_custom)
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO denuncias
              (protocolo, numero_whatsapp, modulo, subcategoria, descricao,
               local, data_ocorrencia, anonimo, foto_url, status)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        """,
            protocolo,
            dados["numero_whatsapp"],
            dados["modulo"],
            dados["subcategoria"],
            dados["descricao"],
            dados["local"],
            dados["data_ocorrencia"],
            dados["anonimo"],
            dados.get("foto_url"),
            dados.get("status", "pendente")
        )
    return protocolo

# ==============================
# 📲 WHATSAPP
# ==============================
async def enviar_mensagem(numero: str, texto: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{WHAPI_URL}/messages/text",
            json={"to": numero, "body": texto},
            headers={"Authorization": f"Bearer {WHAPI_TOKEN}"}
        )

async def enviar_audio(numero: str, audio_b64: str):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{WHAPI_URL}/messages/audio",
            json={
                "to": numero,
                "media": f"data:audio/ogg;base64,{audio_b64}",
                "mime_type": "audio/ogg; codecs=opus"
            },
            headers={"Authorization": f"Bearer {WHAPI_TOKEN}"}
        )
        print(f"[AUDIO] Resposta whapi envio: {r.status_code} {r.text[:200]}")

# ==============================
# 📍 GEOCODING (Nominatim / OpenStreetMap)
# ==============================
async def geocodificar(lat: float, lon: float) -> str:
    """Converte lat/lon em endereco legivel via Nominatim."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"lat": lat, "lon": lon, "format": "json", "addressdetails": 1},
                headers={"User-Agent": "TucaAI-Denuncias/1.0"}
            )
        data = r.json()
        addr = data.get("address", {})

        # Monta endereço completo
        partes = []
        rua = addr.get("road") or addr.get("pedestrian") or addr.get("path")
        numero = addr.get("house_number")
        bairro = addr.get("suburb") or addr.get("neighbourhood") or addr.get("quarter")
        cidade = addr.get("city") or addr.get("town") or addr.get("municipality")

        if rua:
            partes.append(rua + (f", {numero}" if numero else ""))
        if bairro:
            partes.append(bairro)
        if cidade:
            partes.append(cidade)

        if partes:
            return ", ".join(partes)
        # Fallback: display_name do Nominatim
        return data.get("display_name", f"{lat}, {lon}")
    except Exception as e:
        print(f"[GEO] Erro ao geocodificar: {e}")
        return f"{lat}, {lon}"

# ==============================
# 🧠 OLLAMA
# ==============================
async def perguntar_ollama(prompt: str, max_tokens: int = 200) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3}
        })
    return r.json()["response"].strip()

# ==============================
# 🧠 CLASSIFICAÇÃO
# ==============================
# Mapeamento direto de palavras-chave
KEYWORDS = {
    ("alagamento", "alagou", "inundacao", "inundação", "enchente", "alagada"): ("Infraestrutura", "Alagamento"),
    ("buraco", "cratera", "asfalto danificado"): ("Infraestrutura", "Buraco na via"),
    ("calcada", "calçada", "passeio"): ("Infraestrutura", "Calçada danificada"),
    ("iluminacao", "iluminação", "poste", "lampada", "lâmpada"): ("Infraestrutura", "Iluminação pública"),
    ("esgoto", "bueiro", "valeta"): ("Infraestrutura", "Esgoto"),
    ("obra irregular", "construcao irregular"): ("Infraestrutura", "Obra irregular"),
    ("lixo", "entulho", "descarte"): ("Meio Ambiente", "Descarte irregular de lixo"),
    ("poluicao", "poluição", "fumaca", "fumaça"): ("Meio Ambiente", "Poluição"),
    ("desmatamento", "derrubada"): ("Meio Ambiente", "Desmatamento"),
    ("queimada", "incendio", "incêndio"): ("Meio Ambiente", "Queimada"),
    ("animal abandonado", "cao abandonado", "cachorro abandonado"): ("Meio Ambiente", "Animal abandonado"),
    ("vandalismo", "depredacao", "depredação"): ("Dano ao Patrimônio", "Vandalismo"),
    ("pichacao", "pichação", "grafite"): ("Dano ao Patrimônio", "Pichação"),
    ("dengue", "mosquito"): ("Saúde Pública", "Foco de dengue"),
    ("agua contaminada", "água contaminada"): ("Saúde Pública", "Água contaminada"),
    ("semaforo", "semáforo"): ("Mobilidade / Trânsito", "Semáforo com defeito"),
    ("sinalizacao", "sinalização"): ("Mobilidade / Trânsito", "Sinalização apagada"),
    ("acidente"): ("Mobilidade / Trânsito", "Acidente"),
    ("coleta de lixo", "lixeiro"): ("Serviços Públicos", "Falta de coleta de lixo"),
    ("sem agua", "sem água", "falta de agua"): ("Serviços Públicos", "Falta de água"),
    ("sem energia", "sem luz", "falta de energia", "apagao", "apagão"): ("Serviços Públicos", "Falta de energia"),
    ("poda", "galho"): ("Serviços Públicos", "Poda de árvore"),
}

def classificar_por_keyword(mensagem: str):
    import unicodedata
    texto = mensagem.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    for palavras, (modulo, subcategoria) in KEYWORDS.items():
        chaves = (palavras,) if isinstance(palavras, str) else palavras
        if any(p in texto for p in chaves):
            return {"modulo": modulo, "subcategoria": subcategoria, "confianca": "alta"}
    return None

async def classificar(mensagem: str) -> dict:
    resultado_kw = classificar_por_keyword(mensagem)
    if resultado_kw:
        print(f"Classificado por keyword: {resultado_kw}")
        return resultado_kw

    prompt = (
        "Você é um sistema de classificação de denúncias urbanas. "
        "Analise a mensagem e responda APENAS com JSON válido, sem explicações, sem markdown.\n\n"
        "Módulos e subcategorias:\n"
        "- Infraestrutura: Buraco na via, Calçada danificada, Iluminação pública, Esgoto, Alagamento, Obra irregular\n"
        "- Meio Ambiente: Descarte irregular de lixo, Poluição, Desmatamento, Queimada, Animal abandonado\n"
        "- Dano ao Patrimônio: Vandalismo, Pichação, Depredação de bem público\n"
        "- Saúde Pública: Foco de dengue, Esgoto a céu aberto, Estabelecimento irregular, Água contaminada\n"
        "- Mobilidade / Trânsito: Semáforo com defeito, Sinalização apagada, Estacionamento irregular, Acidente\n"
        "- Serviços Públicos: Falta de coleta de lixo, Falta de água, Falta de energia, Poda de árvore\n"
        "- Outros: Outros\n\n"
        "IMPORTANTE: subcategoria deve ser EXATAMENTE uma das listadas acima.\n"
        "- Cumprimento ou mensagem vaga → confianca baixa\n"
        "- Problema claro → confianca alta\n\n"
        f"Mensagem: \"{mensagem}\"\n\n"
        "Responda SOMENTE com este JSON:\n"
        "{\"modulo\":\"...\",\"subcategoria\":\"...\",\"confianca\":\"alta|media|baixa\"}"
    )
    try:
        import json
        resposta = await perguntar_ollama(prompt, max_tokens=80)
        clean = resposta.replace("```json", "").replace("```", "").strip()
        inicio = clean.find("{")
        fim = clean.rfind("}") + 1
        clean = clean[inicio:fim]
        resultado = json.loads(clean)
        if "confiança" in resultado:
            resultado["confianca"] = resultado.pop("confiança")
        if resultado.get("confianca") not in ("alta", "media", "baixa"):
            resultado["confianca"] = "baixa"
        return resultado
    except Exception as e:
        print(f"Erro ao classificar: {e}")
        return {"modulo": "Outros", "subcategoria": "Outros", "confianca": "baixa"}
# ==============================
# 🔎 FILTRO DE RELEVÂNCIA
# ==============================
async def e_denuncia_relevante(mensagem: str) -> bool:
    """Verifica se a mensagem é uma denúncia urbana real ou fora do escopo."""
    prompt = (
        "Você é um filtro para um bot de denúncias urbanas de prefeitura. "
        "Analise a mensagem e responda APENAS com JSON.\n\n"
        "Responda verdadeiro se a mensagem for:\n"
        "- Uma denúncia de problema urbano (buraco, alagamento, lixo, falta de luz, vandalismo, etc)\n"
        "- Um relato de problema na cidade ou bairro\n"
        "- Uma situação que envolva serviços públicos, infraestrutura, meio ambiente ou saúde pública\n\n"
        "Responda falso se for:\n"
        "- Pergunta sobre preços, produtos ou comida\n"
        "- Conversa aleatória, piada ou assunto pessoal\n"
        "- Pedido de informação que não seja sobre problemas urbanos\n"
        "- Cumprimento sem contexto de denúncia\n\n"
        f"Mensagem: \"{mensagem}\"\n\n"
        "Responda SOMENTE: {{\"denuncia\": true}} ou {{\"denuncia\": false}}"
    )
    try:
        import json
        resposta = await perguntar_ollama(prompt, max_tokens=20)
        clean = resposta.replace("```json", "").replace("```", "").strip()
        inicio = clean.find("{")
        fim = clean.rfind("}") + 1
        resultado = json.loads(clean[inicio:fim])
        return resultado.get("denuncia", False)
    except Exception:
        return True  # em caso de dúvida, deixa passar

# ==============================
# 🎙️ PIPELINE DE ÁUDIO
# ==============================
async def transcrever_audio(audio_url: str) -> str | None:
    """Baixa e transcreve o audio, retorna o texto ou None se falhar."""
    try:
        print(f"[AUDIO] Baixando áudio...")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                audio_url,
                headers={"Authorization": f"Bearer {WHAPI_TOKEN}"}
            )
        audio_bytes = resp.content
        print(f"[AUDIO] Áudio baixado: {len(audio_bytes)} bytes | status: {resp.status_code}")

        if resp.status_code != 200 or len(audio_bytes) < 1000:
            print(f"[AUDIO] Resposta raw: {audio_bytes[:300]}")
            return None

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            f.write(audio_bytes)
            tmp_input = f.name

        print("[AUDIO] Transcrevendo com Whisper...")
        try:
            resultado = whisper_model.transcribe(tmp_input, language="pt")
            texto = resultado["text"].strip()
            print(f"[AUDIO] Transcrição: {texto}")
            return texto
        finally:
            os.unlink(tmp_input)

    except Exception as e:
        import traceback
        print(f"[AUDIO] ❌ Erro ao transcrever: {e}")
        traceback.print_exc()
        return None


async def texto_para_audio(texto: str) -> str | None:
    """Converte texto em audio OGG base64 via edge-tts."""
    mp3_path = tempfile.mktemp(suffix=".mp3")
    ogg_path = tempfile.mktemp(suffix=".ogg")
    try:
        print("[AUDIO] Gerando áudio com edge-tts...")
        comunicador = edge_tts.Communicate(texto, VOICE)
        await comunicador.save(mp3_path)

        resultado_ffmpeg = subprocess.run([
            "ffmpeg", "-y", "-i", mp3_path,
            "-c:a", "libopus", "-b:a", "24k", ogg_path
        ], capture_output=True)

        if resultado_ffmpeg.returncode != 0:
            print(f"[AUDIO] Erro ffmpeg: {resultado_ffmpeg.stderr.decode()}")
            return None

        with open(ogg_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[AUDIO] ❌ Erro ao gerar áudio: {e}")
        return None
    finally:
        if os.path.exists(mp3_path): os.unlink(mp3_path)
        if os.path.exists(ogg_path): os.unlink(ogg_path)


async def falar(numero: str, texto: str):
    """Converte texto em audio e envia, sem emojis nem formatacao."""
    audio_b64 = await texto_para_audio(texto)
    if audio_b64:
        await enviar_audio(numero, audio_b64)
    else:
        await enviar_mensagem(numero, texto)


async def responder(numero: str, texto: str):
    """Responde em audio se a sessao preferir, senao em texto."""
    session = sessoes.get(numero, {})
    if session.get("prefere_audio"):
        audio_b64 = await texto_para_audio(texto)
        if audio_b64:
            await enviar_audio(numero, audio_b64)
            return
    await enviar_mensagem(numero, texto)


async def processar_audio(numero: str, audio_url: str):
    texto = await transcrever_audio(audio_url)
    if not texto:
        await falar(numero, "Desculpe, nao consegui processar seu audio. Tente novamente.")
        return

    print(f"[AUDIO] Transcrito: {texto}")

    # ==============================
    # 🚨 EMERGÊNCIA POR ÁUDIO
    # ==============================
    emergencia = detectar_emergencia(texto)
    if emergencia:
        if numero in sessoes:
            del sessoes[numero]
        sessoes[numero] = {
            "etapa":        "AGUARDANDO_LOCAL",
            "descricao":    texto,
            "modulo":       emergencia["modulo"],
            "subcategoria": emergencia["subcategoria"],
            "urgente":      True,
            "sigla":        emergencia["sigla"],
            "prefere_audio": True
        }
        await falar(numero, emergencia["falar"])
        return

    # ==============================
    # 🔍 CONSULTA DE PROTOCOLO POR ÁUDIO
    # ==============================
    import re as _re
    import unicodedata as _ud2

    def _norm_audio(t):
        t = t.lower()
        t = _ud2.normalize("NFD", t)
        return "".join(c for c in t if _ud2.category(c) != "Mn")

    texto_norm = _norm_audio(texto)

    # Detecta intenção de consulta por palavras-chave
    palavras_consulta = ["consultar", "consulta", "protocolo", "status", "acompanhar", "situacao", "situação", "andamento"]
    quer_consultar = any(p in texto_norm for p in palavras_consulta)

    if quer_consultar:
        if numero in sessoes:
            del sessoes[numero]
        # Extrai todos os dígitos da transcrição e monta o protocolo
        digitos = _re.findall(r'\d+', texto)
        numeros_juntos = "".join(digitos)

        # Busca no banco por qualquer protocolo que contenha esses números
        print(f"[AUDIO] Busca por protocolo com dígitos: {numeros_juntos}")
        if numeros_juntos:
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM denuncias WHERE REPLACE(REPLACE(protocolo, '-', ''), ' ', '') LIKE $1 ORDER BY criado_em DESC LIMIT 1",
                    f"%{numeros_juntos}%"
                )
        else:
            row = None

        if row:
            status_map = {
                "pendente":   "pendente, aguardando analise",
                "em_analise": "em analise, sendo apurado",
                "resolvido":  "resolvido, problema tratado",
                "arquivado":  "arquivado",
            }
            status_txt = status_map.get(row["status"], row["status"])
            await falar(numero,
                f"Consulta do protocolo {row['protocolo']}. "
                f"Modulo: {row['modulo']}. "
                f"Tipo: {row['subcategoria']}. "
                f"Local: {row['local']}. "
                f"Status: {status_txt}. "
                f"Se tiver duvidas, entre em contato com a prefeitura.")
        else:
            await falar(numero,
                "Nao encontrei nenhum protocolo com esses numeros. "
                "Verifique o numero e tente novamente, ou envie por texto para maior precisao.")
        return

    # Marca preferência de áudio na sessão
    if numero not in sessoes:
        sessoes[numero] = {}
    sessoes[numero]["prefere_audio"] = True

    # Se já tem sessão com etapa ativa, continua o fluxo de áudio
    if sessoes[numero].get("etapa"):
        await processar_fluxo_audio(numero, texto)
        return

    # Nova mensagem — classifica e inicia o fluxo de áudio
    classificacao = await classificar(texto)

    if classificacao["confianca"] == "baixa":
        sessoes[numero]["etapa"] = "AGUARDANDO_DESCRICAO"
        await falar(numero, "Ola! Sou o Tucu, assistente de denuncias urbanas. Pode me contar o que esta acontecendo?")
    else:
        sessoes[numero].update({
            "etapa":        "AGUARDANDO_LOCAL",
            "descricao":    texto,
            "modulo":       classificacao["modulo"],
            "subcategoria": classificacao["subcategoria"]
        })
        await falar(numero,
            f"Identificamos uma denuncia de {classificacao['modulo']}, sobre {classificacao['subcategoria']}. "
            f"Onde aconteceu? Voce pode compartilhar sua localizacao pelo WhatsApp ou falar o endereco.")

# ==============================
# 📝 FLUXO TEXTO (com emojis e formatação)
# ==============================
async def processar_fluxo(numero: str, mensagem: str):
    session = sessoes[numero]

    if mensagem.strip().lower() == "cancelar":
        del sessoes[numero]
        await enviar_mensagem(numero, "❌ Denúncia cancelada. Se precisar, é só mandar uma nova mensagem.")
        return

    etapa = session["etapa"]

    if etapa == "AGUARDANDO_DESCRICAO":
        classificacao = await classificar(mensagem)
        session["descricao"] = mensagem

        if classificacao["confianca"] == "baixa":
            texto_lower = mensagem.lower()
            perguntas_modulo = ["modulo", "módulo", "categoria", "quais categorias",
                                 "mudar modulo", "trocar modulo", "escolher modulo"]
            if any(p in texto_lower for p in perguntas_modulo):
                await enviar_mensagem(numero,
                    "📂 O módulo é definido automaticamente de acordo com o problema que você descrever — não precisa escolher!\n\n"
                    "Me conte o que está acontecendo e eu identifico a categoria certa. 😊")
            else:
                await enviar_mensagem(numero,
                    "Pode me dar mais detalhes? 🙏\n\nPor exemplo: o que está acontecendo, onde é e há quanto tempo?")
        else:
            session["modulo"] = classificacao["modulo"]
            session["subcategoria"] = classificacao["subcategoria"]
            session["etapa"] = "AGUARDANDO_LOCAL"
            await enviar_mensagem(numero,
                f"Entendi! Vou registrar uma denúncia de:\n"
                f"📂 *{classificacao['modulo']}* › {classificacao['subcategoria']}\n\n"
                f"📍 Onde aconteceu?\n\nCompartilhe sua 📌 *localização pelo WhatsApp* ou digite o endereço (rua, bairro ou ponto de referência)")

    elif etapa == "CONFIRMANDO_CLASSIFICACAO":
        resp = mensagem.strip().lower()
        if resp == "sim":
            session["etapa"] = "AGUARDANDO_LOCAL"
            await enviar_mensagem(numero,
                "📍 Onde aconteceu?\n\nCompartilhe sua 📌 *localização pelo WhatsApp* ou digite o endereço (rua, bairro ou ponto de referência)")
        elif resp in ("não", "nao"):
            session["etapa"] = "CORRIGINDO_CLASSIFICACAO"
            await enviar_mensagem(numero,
                "Sem problema! Me descreva melhor o que está acontecendo que vou identificar novamente.")
        else:
            await enviar_mensagem(numero, "Por favor, responda sim ou não.")

    elif etapa == "CORRIGINDO_CLASSIFICACAO":
        classificacao = await classificar(mensagem)
        session["descricao"] = mensagem

        if classificacao["confianca"] == "baixa":
            await enviar_mensagem(numero,
                "Ainda não consegui identificar o tipo de denúncia. Pode descrever melhor o problema? 🙏")
        else:
            session["modulo"] = classificacao["modulo"]
            session["subcategoria"] = classificacao["subcategoria"]
            session["etapa"] = "AGUARDANDO_LOCAL"
            await enviar_mensagem(numero,
                f"Entendi! Vou registrar uma denúncia de:\n"
                f"📂 *{classificacao['modulo']}* › {classificacao['subcategoria']}\n\n"
                f"📍 Onde aconteceu?\n\nCompartilhe sua 📌 *localização pelo WhatsApp* ou digite o endereço (rua, bairro ou ponto de referência)")

    elif etapa == "AGUARDANDO_LOCAL":
        resp = mensagem.strip().lower()
        if resp in ("sim", "s") and session.get("local"):
            pass  # mantém o local anterior
        else:
            session["local"] = mensagem.strip()
        session["etapa"] = "AGUARDANDO_FOTO"
        await enviar_mensagem(numero,
            "📸 Você tem uma *foto ou vídeo* do problema? (vídeo máx. 30s)\n"
            "Se não tiver, responda *pular*")

    elif etapa == "AGUARDANDO_FOTO":
        session["foto_url"] = None
        session["etapa"] = "CONFIRMANDO_DENUNCIA"
        await enviar_mensagem(numero,
            f"📋 *Resumo da sua denúncia:*\n\n"
            f"📂 Módulo: *{session['modulo']}*\n"
            f"🏷️ Tipo: *{session['subcategoria']}*\n"
            f"📝 Descrição: {session['descricao']}\n"
            f"📍 Local: {session['local']}\n"
            f"📸 Foto: Não\n\n"
            f"Confirma o registro? Responda *sim* ou *não*")

    elif etapa == "CONFIRMANDO_DENUNCIA":
        resp = mensagem.strip().lower()
        if resp in ("sim", "s"):
            protocolo = await registrar_denuncia({
                "numero_whatsapp": numero,
                "modulo":          session["modulo"],
                "subcategoria":    session["subcategoria"],
                "descricao":       session["descricao"],
                "local":           session["local"],
                "data_ocorrencia": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "anonimo":         False,
                "foto_url":        session.get("foto_url"),
                "sigla":           session.get("sigla"),
                "status":          "urgente" if session.get("urgente") else "pendente"
            })
            del sessoes[numero]
            await enviar_mensagem(numero,
                f"✅ *Denúncia registrada com sucesso!*\n\n"
                f"🔖 Protocolo: *{protocolo}*\n\n"
                f"Guarde este número para acompanhar sua denúncia.\nAgradecemos sua contribuição! 🙏")
        elif resp in ("cancelar", "cancela"):
            del sessoes[numero]
            await enviar_mensagem(numero, "❌ Denúncia cancelada. Obrigado!")
        else:
            # Detecta o que a pessoa quer corrigir
            correcao = detectar_correcao(mensagem)
            if correcao == "local":
                session["etapa"] = "AGUARDANDO_LOCAL"
                await enviar_mensagem(numero,
                    "📍 Qual o novo *endereço ou local*?\n(rua, bairro ou ponto de referência)")
            elif correcao == "foto":
                session["etapa"] = "AGUARDANDO_FOTO"
                await enviar_mensagem(numero,
                    "📸 Envie uma *foto ou vídeo* (máx. 30s) ou responda *pular*.")
            else:
                # Quer corrigir o tipo/descrição — extrai descrição limpa e reclassifica
                descricao_limpa = await extrair_descricao_limpa(mensagem)
                classificacao = await classificar(descricao_limpa)
                session["descricao"] = descricao_limpa
                if classificacao["confianca"] != "baixa":
                    session["modulo"] = classificacao["modulo"]
                    session["subcategoria"] = classificacao["subcategoria"]
                session["etapa"] = "CONFIRMANDO_DENUNCIA"
                await enviar_mensagem(numero,
                    f"📋 *Resumo atualizado:*\n\n"
                    f"📂 Módulo: *{session['modulo']}*\n"
                    f"🏷️ Tipo: *{session['subcategoria']}*\n"
                    f"📝 Descrição: {session['descricao']}\n"
                    f"📍 Local: {session['local']}\n"
                    f"📸 Foto: {'Sim' if session.get('foto_url') else 'Não'}\n\n"
                    f"Confirma o registro? Responda *sim* ou *não*")

# ==============================
# 🎙️ FLUXO ÁUDIO (sem emojis, sem formatação)
# ==============================
async def processar_fluxo_audio(numero: str, mensagem: str):
    session = sessoes[numero]

    if normalizar_resposta(mensagem) == "cancelar":
        del sessoes[numero]
        await falar(numero, "Denuncia cancelada. Se precisar, e so mandar uma nova mensagem.")
        return

    etapa = session["etapa"]

    if etapa == "AGUARDANDO_DESCRICAO":
        classificacao = await classificar(mensagem)
        session["descricao"] = mensagem

        if classificacao["confianca"] == "baixa":
            await falar(numero, "Pode me dar mais detalhes? Por exemplo: o que está acontecendo, onde é e há quanto tempo?")
        else:
            session["modulo"] = classificacao["modulo"]
            session["subcategoria"] = classificacao["subcategoria"]
            session["etapa"] = "AGUARDANDO_LOCAL"
            await falar(numero,
                f"Identificamos uma denuncia de {classificacao['modulo']}, sobre {classificacao['subcategoria']}. "
                f"Onde aconteceu? Voce pode compartilhar sua localizacao pelo WhatsApp ou falar o endereco.")

    elif etapa == "CONFIRMANDO_CLASSIFICACAO":
        resp = normalizar_resposta(mensagem)
        if resp == "sim":
            session["etapa"] = "AGUARDANDO_LOCAL"
            await falar(numero, "Onde aconteceu? Voce pode compartilhar sua localizacao pelo WhatsApp ou falar o endereco.")
        elif resp in ("nao", "n"):
            session["etapa"] = "CORRIGINDO_CLASSIFICACAO"
            await falar(numero, "Sem problema. Me descreva melhor o que esta acontecendo que vou identificar novamente.")
        else:
            await falar(numero, "Por favor, responda sim ou nao.")

    elif etapa == "CORRIGINDO_CLASSIFICACAO":
        classificacao = await classificar(mensagem)
        session["descricao"] = mensagem

        if classificacao["confianca"] == "baixa":
            await falar(numero, "Ainda não consegui identificar o tipo de denúncia. Pode descrever melhor o problema?")
        else:
            session["modulo"] = classificacao["modulo"]
            session["subcategoria"] = classificacao["subcategoria"]
            session["etapa"] = "AGUARDANDO_LOCAL"
            await falar(numero,
                f"Identificamos uma denuncia de {classificacao['modulo']}, sobre {classificacao['subcategoria']}. "
                f"Onde aconteceu? Voce pode compartilhar sua localizacao pelo WhatsApp ou falar o endereco.")

    elif etapa == "AGUARDANDO_LOCAL":
        resp = normalizar_resposta(mensagem)
        if resp == "sim" and session.get("local"):
            pass  # mantém o local anterior
        else:
            session["local"] = mensagem.strip()
        session["etapa"] = "AGUARDANDO_FOTO"
        await falar(numero, "Voce tem uma foto ou video curto do problema? Video de no maximo 30 segundos. Se nao tiver, diga pular.")

    elif etapa == "AGUARDANDO_FOTO":
        session["foto_url"] = None
        session["etapa"] = "CONFIRMANDO_DENUNCIA"
        await falar(numero,
            f"Resumo da sua denuncia. "
            f"Modulo: {session['modulo']}. "
            f"Tipo: {session['subcategoria']}. "
            f"Descricao: {session['descricao']}. "
            f"Local: {session['local']}. "
            f"Foto: nao. "
            f"Confirma o registro? Responda sim ou nao.")

    elif etapa == "CONFIRMANDO_DENUNCIA":
        resp = normalizar_resposta(mensagem)
        if resp == "sim":
            protocolo = await registrar_denuncia({
                "numero_whatsapp": numero,
                "modulo":          session["modulo"],
                "subcategoria":    session["subcategoria"],
                "descricao":       session["descricao"],
                "local":           session["local"],
                "data_ocorrencia": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "anonimo":         False,
                "foto_url":        session.get("foto_url")
            })
            del sessoes[numero]
            await falar(numero,
                f"Denuncia registrada com sucesso! "
                f"Seu protocolo e {protocolo}. "
                f"Guarde este numero para acompanhar sua denuncia. Obrigado!")
        elif resp in ("cancelar", "cancela"):
            del sessoes[numero]
            await falar(numero, "Denuncia cancelada. Obrigado!")
        else:
            correcao = detectar_correcao(mensagem)
            if correcao == "local":
                session["etapa"] = "AGUARDANDO_LOCAL"
                await falar(numero, "Qual o novo endereco ou local?")
            elif correcao == "foto":
                session["etapa"] = "AGUARDANDO_FOTO"
                await falar(numero, "Envie uma foto ou video de ate 30 segundos, ou diga pular.")
            else:
                descricao_limpa = await extrair_descricao_limpa(mensagem)
                classificacao = await classificar(descricao_limpa)
                session["descricao"] = descricao_limpa
                if classificacao["confianca"] != "baixa":
                    session["modulo"] = classificacao["modulo"]
                    session["subcategoria"] = classificacao["subcategoria"]
                session["etapa"] = "CONFIRMANDO_DENUNCIA"
                await falar(numero,
                    f"Resumo atualizado. "
                    f"Modulo: {session['modulo']}. "
                    f"Tipo: {session['subcategoria']}. "
                    f"Descricao: {session['descricao']}. "
                    f"Local: {session['local']}. "
                    f"Foto: {'sim' if session.get('foto_url') else 'nao'}. "
                    f"Confirma o registro? Responda sim ou nao.")

# ==============================
# 🔗 WEBHOOK
# ==============================
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()

    messages = body.get("messages", [])
    if not messages:
        return {"status": "ok"}

    msg = messages[0]
    if msg.get("from_me"):
        return {"status": "ok"}

    numero = msg.get("from")

    tipo = msg.get("type")

    # 📍 LOCALIZAÇÃO — converte lat/lon em endereço
    if tipo == "location":
        if numero in sessoes and sessoes[numero].get("etapa") == "AGUARDANDO_LOCAL":
            loc = msg.get("location") or {}
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            if lat and lon:
                print(f"[GEO] Localização recebida: {lat}, {lon}")
                endereco = await geocodificar(float(lat), float(lon))
                print(f"[GEO] Endereço: {endereco}")
                sessoes[numero]["local"] = endereco
                sessoes[numero]["etapa"] = "AGUARDANDO_FOTO"
                prefere_audio = sessoes[numero].get("prefere_audio")
                if prefere_audio:
                    await falar(numero,
                        f"Localizacao recebida. Endereco identificado: {endereco}. "
                        f"Voce tem uma foto ou video de ate 30 segundos? Se nao, diga pular.")
                else:
                    await enviar_mensagem(numero,
                        f"📍 Localização recebida!\n"
                        f"*Endereço identificado:* {endereco}\n\n"
                        f"📸 Você tem uma *foto ou vídeo* do problema? (vídeo máx. 30s)\n"
                        f"Se não tiver, responda *pular*")
        return {"status": "ok"}

    # 🖼️ IMAGEM — salva na sessão se estiver aguardando foto
    if tipo in ("image", "video"):
        if numero in sessoes and sessoes[numero].get("etapa") == "AGUARDANDO_FOTO":
            image_data = msg.get("image") or msg.get("video") or {}
            import json as _json
            preview   = image_data.get("preview")
            media_id  = image_data.get("id")
            mime_type = image_data.get("mime_type", "")
            print(f"[FOTO] Midia recebida. ID: {media_id} | tipo: {mime_type}")
            foto_url = _json.dumps({"preview": preview, "id": media_id, "mime_type": mime_type}) if (preview or media_id) else None
            sessoes[numero]["foto_url"] = foto_url
            prefere_audio = sessoes[numero].get("prefere_audio")
            # Avança para confirmação
            session = sessoes[numero]
            session["foto_url"] = foto_url
            session["etapa"] = "CONFIRMANDO_DENUNCIA"
            if prefere_audio:
                await falar(numero,
                    f"Foto recebida! "
                    f"Resumo da sua denuncia. "
                    f"Modulo: {session['modulo']}. "
                    f"Tipo: {session['subcategoria']}. "
                    f"Descricao: {session['descricao']}. "
                    f"Local: {session['local']}. "
                    f"Foto: sim. "
                    f"Confirma o registro? Responda sim ou nao.")
            else:
                await enviar_mensagem(numero,
                    f"📸 Foto recebida!\n\n"
                    f"📋 *Resumo da sua denúncia:*\n\n"
                    f"📂 Módulo: *{session['modulo']}*\n"
                    f"🏷️ Tipo: *{session['subcategoria']}*\n"
                    f"📝 Descrição: {session['descricao']}\n"
                    f"📍 Local: {session['local']}\n"
                    f"📸 Foto: Sim\n\n"
                    f"Confirma o registro? Responda *sim* ou *não*")
        return {"status": "ok"}

    # 🎙️ ÁUDIO
    if tipo in ("audio", "voice"):
        print(f"Áudio recebido de {numero}")
        voice_data = msg.get("audio") or msg.get("voice") or {}
        audio_id = voice_data.get("id")
        if audio_id:
            audio_url = f"https://gate.whapi.cloud/media/{audio_id}"
            print(f"[AUDIO] URL montada: {audio_url}")
            background_tasks.add_task(processar_audio, numero, audio_url)
        else:
            print("[AUDIO] ❌ ID de áudio não encontrado no payload!")
        return {"status": "ok"}

    # 💬 TEXTO
    texto = (msg.get("text") or {}).get("body")
    if not texto:
        return {"status": "ok"}

    print(f"Mensagem de {numero}: {texto}")

    # ==============================
    # 🚨 DETECÇÃO DE EMERGÊNCIA — máxima prioridade
    # ==============================
    emergencia = detectar_emergencia(texto)
    if emergencia:
        # Cancela sessão se houver
        if numero in sessoes:
            del sessoes[numero]
        # Inicia sessão de emergência direto no local
        sessoes[numero] = {
            "etapa":        "AGUARDANDO_LOCAL",
            "descricao":    texto,
            "modulo":       emergencia["modulo"],
            "subcategoria": emergencia["subcategoria"],
            "urgente":      True,
            "sigla":        emergencia["sigla"]
        }
        await enviar_mensagem(numero, emergencia["resposta"])
        return {"status": "ok"}

    # ==============================
    # 🔍 CONSULTA DE PROTOCOLO — verifica ANTES da sessão
    # ==============================
    import re as _re
    protocolo_match = _re.search(r'\b([A-Z]{2,4}-\d{2,4}-\d{2,4})\b', texto.upper())
    if protocolo_match:
        # Cancela sessão ativa se houver (pessoa desistiu pra consultar)
        if numero in sessoes:
            del sessoes[numero]
        protocolo_buscado = protocolo_match.group(1)
        print(f"Consulta de protocolo: {protocolo_buscado}")
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM denuncias WHERE UPPER(protocolo) = $1", protocolo_buscado
            )
        if row:
            status_map = {
                "pendente":   "🟡 Pendente — aguardando análise",
                "em_analise": "🔵 Em análise — sendo apurado",
                "resolvido":  "🟢 Resolvido — problema tratado",
                "arquivado":  "⚫ Arquivado",
            }
            status_txt = status_map.get(row["status"], row["status"])
            await enviar_mensagem(numero,
                f"🔍 *Consulta de protocolo*\n\n"
                f"🔖 Protocolo: *{row['protocolo']}*\n"
                f"📂 Módulo: {row['modulo']}\n"
                f"🏷️ Tipo: {row['subcategoria']}\n"
                f"📍 Local: {row['local']}\n"
                f"📅 Registrado em: {row['data_ocorrencia']}\n"
                f"📊 Status: {status_txt}\n\n"
                f"Se tiver dúvidas, entre em contato com a prefeitura.")
        else:
            await enviar_mensagem(numero,
                f"❌ Protocolo *{protocolo_buscado}* não encontrado.\n"
                f"Verifique o número e tente novamente.")
        return {"status": "ok"}

    if numero in sessoes:
        await processar_fluxo(numero, texto)
        return {"status": "ok"}

    # ==============================
    # 👋 MENSAGENS DE ENCERRAMENTO — ignora
    # ==============================
    import unicodedata as _ud
    def _norm(t):
        t = t.lower().strip()
        t = _ud.normalize("NFD", t)
        return "".join(c for c in t if _ud.category(c) != "Mn")

    # ==============================
    # ❓ PERGUNTAS SOBRE OS CAMPOS
    # ==============================
    perguntas_campos = {
        ("o que e modulo", "o que é modulo", "o que significa modulo", "pra que serve modulo", "modulo e o que"): (
            "📂 *Módulo* é a grande área da sua denúncia. Exemplo: Infraestrutura, Meio Ambiente, Saúde Pública, etc.\n\n"
            "Ele é identificado automaticamente pela IA de acordo com o que você descrever. Não precisa escolher!"
        ),
        ("o que e tipo", "o que é tipo", "o que significa tipo", "pra que serve tipo", "tipo e o que", "subcategoria"): (
            "🏷️ *Tipo* é a subcategoria específica dentro do módulo. Exemplo: dentro de Infraestrutura, o tipo pode ser *Alagamento*, *Buraco na via*, *Iluminação pública*, etc.\n\n"
            "Também é definido automaticamente pela IA!"
        ),
        ("o que e local", "o que é local", "o que significa local", "pra que serve local", "local e o que"): (
            "📍 *Local* é o endereço ou ponto de referência onde o problema está ocorrendo. Exemplo: *Rua das Flores, próximo ao mercado X*."
        ),
        ("o que e foto", "o que é foto", "pra que serve foto", "foto e obrigatorio", "foto é obrigatorio"): (
            "📸 *Foto* é opcional! Se você tiver uma imagem do problema, ela ajuda muito na apuração da denúncia. Mas pode pular se não tiver."
        ),
        ("o que e protocolo", "o que é protocolo", "pra que serve protocolo", "protocolo e o que"): (
            "🔖 *Protocolo* é o número único gerado após o registro da sua denúncia. Com ele você pode acompanhar o status — é só me enviar o número quando quiser consultar!"
        ),
        ("o que e descricao", "o que é descricao", "o que é descrição", "descricao e o que", "descrição é o que"): (
            "📝 *Descrição* é o relato do problema que você me enviou. É o que você contou sobre o que está acontecendo."
        ),
        ("o que e status", "o que é status", "status e o que", "o que significa status"): (
            "📊 *Status* indica em que fase está sua denúncia:\n\n"
            "🟡 *Pendente* — aguardando análise\n"
            "🔵 *Em análise* — sendo apurado\n"
            "🟢 *Resolvido* — problema tratado\n"
            "⚫ *Arquivado* — encerrado sem resolução"
        ),
    }

    texto_norm_campo = _norm(texto)
    for chaves, resposta_campo in perguntas_campos.items():
        if any(c in texto_norm_campo for c in chaves):
            await enviar_mensagem(numero, resposta_campo)
            return {"status": "ok"}

    _encerramentos = [
        "ok", "obrigado", "obrigada", "valeu", "vlw", "tmj", "ok obrigado",
        "ok obrigada", "ok valeu", "entendido", "certo", "tudo bem",
        "perfeito", "otimo", "ótimo", "show", "blz", "beleza", "ate mais",
        "tchau", "adeus", "flw", "falou", "boa noite", "bom dia", "boa tarde",
        "ok vlw", "ok valeu", "muito obrigado", "muito obrigada"
    ]
    if _norm(texto) in _encerramentos:
        return {"status": "ok"}


    # Filtro de relevância — verifica se é denúncia real antes de classificar
    relevante = await e_denuncia_relevante(texto)
    if not relevante:
        texto_lower = texto.lower()
        perguntas_bot = ["o que voce faz", "o que você faz", "pra que serve", "para que serve",
                         "como funciona", "o que e isso", "o que é isso", "me ajuda", "ajuda",
                         "oi", "ola", "olá", "bom dia", "boa tarde", "boa noite", "hey", "hi"]
        perguntas_modulo = ["modulo", "módulo", "categoria", "quais categorias",
                             "quais modulos", "quais módulos", "mudar modulo", "trocar modulo"]

        if any(p in texto_lower for p in perguntas_modulo):
            await enviar_mensagem(numero,
                "📂 *Sobre os módulos de denúncia*\n\n"
                "Os módulos são definidos automaticamente pela IA de acordo com o problema que você descrever. Não é necessário escolher!\n\n"
                "Basta me contar o que está acontecendo e eu identifico a categoria correta:\n\n"
                "🏗️ *Infraestrutura* — buracos, alagamentos, iluminação...\n"
                "🌿 *Meio Ambiente* — lixo irregular, poluição...\n"
                "🏛️ *Dano ao Patrimônio* — pichação, vandalismo...\n"
                "🏥 *Saúde Pública* — dengue, esgoto a céu aberto...\n"
                "🚗 *Mobilidade / Trânsito* — semáforo, sinalização...\n"
                "⚙️ *Serviços Públicos* — falta de água, luz, coleta...\n\n"
                "Me conte o problema e cuido do resto! 😊")
        elif any(p in texto_lower for p in perguntas_bot):
            sessoes[numero] = {"etapa": "AGUARDANDO_DESCRICAO"}
            await enviar_mensagem(numero,
                "Olá! 👋 Sou o *Tucu*, assistente de denúncias urbanas.\n\n"
                "Estou aqui para registrar problemas da sua cidade como:\n"
                "• Buracos e alagamentos\n"
                "• Falta de luz ou água\n"
                "• Lixo irregular\n"
                "• Pichação e vandalismo\n"
                "• E muito mais!\n\n"
                "Me conte o que está acontecendo que registro sua denúncia. 🙏")
        else:
            # Limpa sessão se existir
            if numero in sessoes:
                del sessoes[numero]
            await enviar_mensagem(numero,
                "Sou o *Tucu* — registro apenas problemas de infraestrutura e serviços públicos da cidade, como:\n\n"
                "Sou o *Tucu* — registro apenas problemas de infraestrutura e serviços públicos da cidade, como:\n\n"
                "🕳️ Buracos e alagamentos\n"
                "💡 Falta de energia ou iluminação\n"
                "💧 Falta de água\n"
                "🗑️ Descarte irregular de lixo\n"
                "🏚️ Vandalismo e pichação\n"
                "🦟 Focos de dengue\n"
                "🚦 Problemas no trânsito\n\n"
                "Se tiver algum desses problemas para denunciar, é só me contar! 🙏")
        return {"status": "ok"}

    classificacao = await classificar(texto)

    if classificacao["confianca"] == "baixa":
        sessoes[numero] = {"etapa": "AGUARDANDO_DESCRICAO"}
        await enviar_mensagem(numero,
            "Olá! 👋 Sou o *Tucu*, assistente de denúncias urbanas.\n\nPode me contar o problema que deseja denunciar?")
    else:
        sessoes[numero] = {
            "etapa":        "AGUARDANDO_LOCAL",
            "descricao":    texto,
            "modulo":       classificacao["modulo"],
            "subcategoria": classificacao["subcategoria"]
        }
        await enviar_mensagem(numero,
            f"Entendi! Vou registrar uma denúncia de:\n"
            f"📂 *{classificacao['modulo']}* › {classificacao['subcategoria']}\n\n"
            f"📍 Onde aconteceu?\n\nCompartilhe sua 📌 *localização pelo WhatsApp* ou digite o endereço (rua, bairro ou ponto de referência)")

    return {"status": "ok"}