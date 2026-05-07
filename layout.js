async function renderLayout(pageTitle, pageSub, activeNav) {
  const me = await fetch('/api/me').then(r => r.json()).catch(() => null);
  if (!me) { window.location.href = '/login'; return; }

  const inicial = me.usuario.charAt(0).toUpperCase();

  document.body.innerHTML = `
    <div class="layout">
      <aside class="sidebar">
        <div class="sidebar-logo">
          <div class="brand">Tuca<span>AI</span></div>
          <div class="subtitle">Sistema de Denúncias</div>
        </div>
        <nav class="sidebar-nav">
          <a href="/" class="nav-item ${activeNav === 'dashboard' ? 'active' : ''}">
            <span class="icon">📊</span> Dashboard
          </a>
          <a href="/denuncias" class="nav-item ${activeNav === 'denuncias' ? 'active' : ''}">
            <span class="icon">📋</span> Denúncias
          </a>
        </nav>
        <div class="sidebar-footer">
          <div class="user-info">
            <div class="user-avatar">${inicial}</div>
            <div>
              <div class="user-name">${me.usuario}</div>
              <div class="user-role">Administrador</div>
            </div>
            <form action="/logout" method="POST" style="margin-left:auto">
              <button class="btn-logout" title="Sair">⎋</button>
            </form>
          </div>
        </div>
      </aside>

      <div class="main">
        <div class="topbar">
          <div>
            <div class="page-title">${pageTitle}</div>
            <div class="page-sub">${pageSub}</div>
          </div>
        </div>
        <div class="content" id="content"></div>
      </div>
    </div>

    <div class="toast" id="toast"></div>
  `;
}

function toast(msg, type = 'success') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast ${type} show`;
  setTimeout(() => t.className = 'toast', 3000);
}

function badgeHTML(status) {
  const map = {
    pendente:   '🟡 Pendente',
    em_analise: '🔵 Em análise',
    resolvido:  '🟢 Resolvido',
    arquivado:  '⚫ Arquivado'
  };
  return `<span class="badge badge-${status}">${map[status] || status}</span>`;
}

function formatDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('pt-BR', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}