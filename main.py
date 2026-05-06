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

async def extrair_descricao_limpa(mensagem: str) -> str:
    """Remove preâmbulos como 'errei, na verdade' e extrai só a descrição real."""
    prompt = (
        "Extraia apenas a descrição do problema da mensagem abaixo, removendo qualquer preâmbulo "
        "como 'errei', 'na verdade', 'gostaria de mudar para', 'quero alterar', etc.\n"
        "Retorne SOMENTE a descrição limpa, sem aspas, sem explicações.\n\n"
        f"Mensagem: {mensagem}\n\n"
        "Descrição limpa:"
    )
    try:
        resultado = await perguntar_ollama(prompt, max_tokens=80)
        # Remove aspas se vier com elas
        limpo = resultado.strip().strip('"').strip("'")
        return limpo if limpo else mensagem
    except Exception:
        return mensagem

def detectar_correcao(mensagem: str) -> str | None:
    """Detecta o que a pessoa quer corrigir. Retorna: 'tipo', 'local', 'foto' ou None."""
    import unicodedata
    texto = mensagem.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")

    if any(p in texto for p in ["local", "endereco", "endereço", "rua", "bairro", "lugar", "logradouro"]):
        return "local"
    if any(p in texto for p in ["foto", "imagem", "print", "fotografia", "picture"]):
        return "foto"
    # Se descreve outro problema = quer corrigir o tipo
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
    whisper_model = whisper.load_model("small", device="cuda")
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
                status           VARCHAR(20)  DEFAULT 'pendente',
                criado_em        TIMESTAMP    DEFAULT NOW()
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_denuncias_status    ON denuncias(status)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_denuncias_modulo    ON denuncias(modulo)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_denuncias_criado_em ON denuncias(criado_em)")

def gerar_protocolo():
    ano = datetime.now().year
    num = random.randint(10000, 99999)
    return f"DEN-{ano}-{num}"

async def registrar_denuncia(dados: dict) -> str:
    protocolo = gerar_protocolo()
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
            "pendente"
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
# 🎙️ PIPELINE DE ÁUDIO
# ==============================
async def transcrever_audio(audio_url: str) -> str | None:
    """Baixa e transcreve o áudio, retorna o texto ou None se falhar."""
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
    """Converte texto em áudio OGG base64 via edge-tts."""
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
    """Converte texto em audio e envia — sem emojis nem formatacao."""
    audio_b64 = await texto_para_audio(texto)
    if audio_b64:
        await enviar_audio(numero, audio_b64)
    else:
        await enviar_mensagem(numero, texto)


async def responder(numero: str, texto: str):
    """Responde em áudio se a sessão preferir, senão em texto."""
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
        await falar(numero, "Ola! Sou o assistente de denuncias. Pode me contar o que esta acontecendo?")
    else:
        sessoes[numero].update({
            "etapa":        "CONFIRMANDO_CLASSIFICACAO",
            "descricao":    texto,
            "modulo":       classificacao["modulo"],
            "subcategoria": classificacao["subcategoria"]
        })
        await falar(numero,
            f"Identificamos uma denuncia de {classificacao['modulo']}, sobre {classificacao['subcategoria']}. "
            f"Qual o endereco ou local onde aconteceu?")

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
            await enviar_mensagem(numero,
                "Pode me dar mais detalhes? 🙏\n\nPor exemplo: o que está acontecendo, onde é e há quanto tempo?")
        else:
            session["modulo"] = classificacao["modulo"]
            session["subcategoria"] = classificacao["subcategoria"]
            session["etapa"] = "AGUARDANDO_LOCAL"
            await enviar_mensagem(numero,
                f"Entendi! Vou registrar uma denúncia de:\n"
                f"📂 *{classificacao['modulo']}* › {classificacao['subcategoria']}\n\n"
                f"📍 Qual o *endereço ou local* onde aconteceu?\n(rua, bairro ou ponto de referência)")

    elif etapa == "CONFIRMANDO_CLASSIFICACAO":
        resp = mensagem.strip().lower()
        if resp == "sim":
            session["etapa"] = "AGUARDANDO_LOCAL"
            await enviar_mensagem(numero,
                "📍 Qual o endereço ou local onde aconteceu?\n(rua, bairro ou ponto de referência)")
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
                f"📍 Qual o *endereço ou local* onde aconteceu?\n(rua, bairro ou ponto de referência)")

    elif etapa == "AGUARDANDO_LOCAL":
        resp = mensagem.strip().lower()
        if resp in ("sim", "s") and session.get("local"):
            pass  # mantém o local anterior
        else:
            session["local"] = mensagem.strip()
        session["etapa"] = "AGUARDANDO_FOTO"
        await enviar_mensagem(numero,
            "📸 Você tem uma foto do problema? Se sim, envie agora.\n"
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
                "foto_url":        session.get("foto_url")
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
                    "📸 Envie a nova foto ou responda *pular* para continuar sem foto.")
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
                f"Qual o endereco ou local onde aconteceu?")

    elif etapa == "CONFIRMANDO_CLASSIFICACAO":
        resp = normalizar_resposta(mensagem)
        if resp == "sim":
            session["etapa"] = "AGUARDANDO_LOCAL"
            await falar(numero, "Qual o endereco ou local onde aconteceu? Pode falar o nome da rua, bairro ou um ponto de referencia.")
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
                f"Qual o endereco ou local onde aconteceu?")

    elif etapa == "AGUARDANDO_LOCAL":
        resp = normalizar_resposta(mensagem)
        if resp == "sim" and session.get("local"):
            pass  # mantém o local anterior
        else:
            session["local"] = mensagem.strip()
        session["etapa"] = "AGUARDANDO_FOTO"
        await falar(numero, "Voce tem uma foto do problema? Se sim, envie agora. Se nao tiver, diga pular.")

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
                await falar(numero, "Envie a nova foto ou diga pular para continuar sem foto.")
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

    # 🖼️ IMAGEM — salva na sessão se estiver aguardando foto
    tipo = msg.get("type")
    if tipo == "image":
        if numero in sessoes and sessoes[numero].get("etapa") == "AGUARDANDO_FOTO":
            image_data = msg.get("image") or {}
            foto_url = image_data.get("link") or image_data.get("id") or "recebida"
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

    if numero in sessoes:
        await processar_fluxo(numero, texto)
        return {"status": "ok"}

    classificacao = await classificar(texto)

    if classificacao["confianca"] == "baixa":
        sessoes[numero] = {"etapa": "AGUARDANDO_DESCRICAO"}
        await responder(numero,
            "Olá! 👋 Sou o assistente de denúncias.\n\nPode me contar o que está acontecendo?")
    else:
        sessoes[numero] = {
            "etapa":        "CONFIRMANDO_CLASSIFICACAO",
            "descricao":    texto,
            "modulo":       classificacao["modulo"],
            "subcategoria": classificacao["subcategoria"]
        }
        await enviar_mensagem(numero,
            f"Entendi! Vou registrar uma denúncia de:\n"
            f"📂 *{classificacao['modulo']}* › {classificacao['subcategoria']}\n\n"
            f"📍 Qual o *endereço ou local* onde aconteceu?\n(rua, bairro ou ponto de referência)")

    return {"status": "ok"}