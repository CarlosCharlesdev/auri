require('dotenv').config();
const express = require('express');
const session = require('express-session');
const { Pool } = require('pg');
const path = require('path');

const app = express();
const PORT = process.env.DASHBOARD_PORT || 4000;

// DB
const pool = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     process.env.DB_PORT     || 5432,
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'ia_vendas'
});

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(__dirname));
app.use(session({
  secret: process.env.SESSION_SECRET || 'zapai-secret-2026',
  resave: false,
  saveUninitialized: false,
  cookie: { maxAge: 8 * 60 * 60 * 1000 } // 8 horas
}));

// Auth middleware
const auth = (req, res, next) => {
  if (req.session.user) return next();
  res.redirect('/login');
};

// ==============================
// LOGIN
// ==============================
app.get('/login', (req, res) => {
  if (req.session.user) return res.redirect('/');
  res.sendFile(path.join(__dirname, 'login.html'));
});

app.post('/login', (req, res) => {
  const { usuario, senha } = req.body;
  const users = JSON.parse(process.env.USERS || '[{"usuario":"admin","senha":"admin123"}]');
  const user = users.find(u => u.usuario === usuario && u.senha === senha);
  if (user) {
    req.session.user = { usuario: user.usuario, nome: user.nome || usuario };
    res.json({ ok: true });
  } else {
    res.status(401).json({ ok: false, erro: 'Usuário ou senha inválidos' });
  }
});

app.post('/logout', (req, res) => {
  req.session.destroy();
  res.redirect('/login');
});

// ==============================
// PÁGINAS
// ==============================
app.get('/', auth, (req, res) => res.sendFile(path.join(__dirname, 'index.html')));
app.get('/denuncias', auth, (req, res) => res.sendFile(path.join(__dirname, 'denuncias.html')));
app.get('/denuncia/:id', auth, (req, res) => res.sendFile(path.join(__dirname, 'detalhe.html')));

// ==============================
// API
// ==============================

// Stats para o dashboard
app.get('/api/stats', auth, async (req, res) => {
  try {
    const [total, porStatus, porModulo, recentes] = await Promise.all([
      pool.query('SELECT COUNT(*) as total FROM denuncias'),
      pool.query(`SELECT status, COUNT(*) as total FROM denuncias GROUP BY status ORDER BY total DESC`),
      pool.query(`SELECT modulo, COUNT(*) as total FROM denuncias GROUP BY modulo ORDER BY total DESC`),
      pool.query(`SELECT COUNT(*) as total FROM denuncias WHERE criado_em >= NOW() - INTERVAL '7 days'`)
    ]);

    res.json({
      total: parseInt(total.rows[0].total),
      porStatus: porStatus.rows,
      porModulo: porModulo.rows,
      ultimos7dias: parseInt(recentes.rows[0].total)
    });
  } catch (e) {
    res.status(500).json({ erro: e.message });
  }
});

// Lista de denúncias com filtros
app.get('/api/denuncias', auth, async (req, res) => {
  try {
    const { modulo, status, data_inicio, data_fim, busca, page = 1 } = req.query;
    const limit = 20;
    const offset = (page - 1) * limit;

    let where = [];
    let params = [];
    let i = 1;

    if (modulo)      { where.push(`modulo = $${i++}`);                          params.push(modulo); }
    if (status)      { where.push(`status = $${i++}`);                          params.push(status); }
    if (data_inicio) { where.push(`criado_em >= $${i++}`);                      params.push(data_inicio); }
    if (data_fim)    { where.push(`criado_em <= $${i++}::date + interval '1 day'`); params.push(data_fim); }
    if (busca)       { where.push(`(descricao ILIKE $${i} OR local ILIKE $${i} OR protocolo ILIKE $${i})`); params.push(`%${busca}%`); i++; }

    const whereStr = where.length ? `WHERE ${where.join(' AND ')}` : '';

    const [rows, count] = await Promise.all([
      pool.query(`SELECT * FROM denuncias ${whereStr} ORDER BY criado_em DESC LIMIT ${limit} OFFSET ${offset}`, params),
      pool.query(`SELECT COUNT(*) as total FROM denuncias ${whereStr}`, params)
    ]);

    res.json({ denuncias: rows.rows, total: parseInt(count.rows[0].total), page: parseInt(page), pages: Math.ceil(count.rows[0].total / limit) });
  } catch (e) {
    res.status(500).json({ erro: e.message });
  }
});

// Detalhe de uma denúncia
app.get('/api/denuncia/:id', auth, async (req, res) => {
  try {
    const { rows } = await pool.query('SELECT * FROM denuncias WHERE id = $1', [req.params.id]);
    if (!rows.length) return res.status(404).json({ erro: 'Não encontrada' });
    res.json(rows[0]);
  } catch (e) {
    res.status(500).json({ erro: e.message });
  }
});

// Atualizar status
app.patch('/api/denuncia/:id/status', auth, async (req, res) => {
  try {
    const { status } = req.body;
    const validos = ['pendente', 'em_analise', 'resolvido', 'arquivado'];
    if (!validos.includes(status)) return res.status(400).json({ erro: 'Status inválido' });
    await pool.query('UPDATE denuncias SET status = $1 WHERE id = $2', [status, req.params.id]);
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ erro: e.message });
  }
});

// Proxy de imagem do whapi — serve a foto pro browser autenticado
app.get('/api/foto/:id', auth, async (req, res) => {
  try {
    const https = require('https');
    const token = process.env.WHATSAPP_TOKEN || '';
    // foto_url pode ser: ID puro, URL completa do whapi, ou "recebida"
    const raw = decodeURIComponent(req.params.id);
    let fetchUrl;
    if (raw.startsWith('http')) {
      fetchUrl = raw; // URL completa — usa direto
    } else {
      fetchUrl = `https://gate.whapi.cloud/media/${raw}`; // só o ID
    }
    const url = fetchUrl;

    const request = https.get(url, { headers: { Authorization: `Bearer ${token}` } }, (upstream) => {
      if (upstream.statusCode !== 200) {
        return res.status(404).send('Foto não encontrada');
      }
      res.setHeader('Content-Type', upstream.headers['content-type'] || 'image/jpeg');
      res.setHeader('Cache-Control', 'private, max-age=3600');
      upstream.pipe(res);
    });

    request.on('error', () => res.status(500).send('Erro ao buscar imagem'));
  } catch (e) {
    res.status(500).send(e.message);
  }
});

// Usuário logado
app.get('/api/me', auth, (req, res) => res.json(req.session.user));

app.listen(PORT, () => console.log(`🌐 Dashboard rodando em http://localhost:${PORT}`));