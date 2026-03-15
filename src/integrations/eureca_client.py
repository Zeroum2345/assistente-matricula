# src/integrations/eureca_client.py
# Cliente da API Eureca — UFCG
#
# A API Eureca (eureca.lsd.ufcg.edu.br/das/v2) é o sistema de dados acadêmicos
# da UFCG. A autenticação é feita com credenciais do SIGAA, que retornam um
# token JWT Bearer usado nas demais chamadas.
#
# Autenticação:
#   POST https://eureca.lsd.ufcg.edu.br/autenticador/sigaa/
#   → retorna token JWT
#   Todas as demais chamadas: Authorization: Bearer <token>
#
# Endpoints utilizados (conforme Swagger em /das/v2/swagger-ui/index.html):
#   GET /das/v2/curriculo/{curriculo}/componentes         → disciplinas do currículo
#   GET /das/v2/curriculo/{curriculo}/componente/{codigo} → detalhes + pré-requisitos
#   GET /das/v2/turmas?periodo=2025.1&componente=COMP3501 → turmas com horários
#   GET /das/v2/aluno/{matricula}/historico               → histórico do aluno (requer aluno autenticado)
#   GET /das/v2/aluno/{matricula}/vinculo                 → vínculo ativo do aluno
#
# Segurança:
#   - Credenciais NUNCA são logadas
#   - Token é armazenado em memória (não em disco)
#   - Dados pessoais de alunos só são acessados com token do próprio aluno
#   - Cache com TTL para reduzir chamadas à API

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

EURECA_BASE_URL  = os.getenv("EURECA_BASE_URL",  "https://eureca.lsd.ufcg.edu.br")
EURECA_AUTH_URL  = os.getenv("EURECA_AUTH_URL",  f"{EURECA_BASE_URL}/autenticador/sigaa/api/login")
EURECA_API_URL   = os.getenv("EURECA_API_URL",   f"{EURECA_BASE_URL}/das/v2")
EURECA_TIMEOUT   = int(os.getenv("EURECA_TIMEOUT", "15"))        # segundos por request
EURECA_CACHE_TTL = int(os.getenv("EURECA_CACHE_TTL", "3600"))    # 1h de cache para dados estáticos

# Credenciais opcionais (para modo "sistema" sem autenticação do aluno)
# Deixe em branco se quiser autenticar sob demanda por aluno
EURECA_LOGIN  = os.getenv("EURECA_LOGIN",  "")
EURECA_PASSWD = os.getenv("EURECA_PASSWD", "")


# ---------------------------------------------------------------------------
# Cache simples em memória (TTL por chave)
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class _TTLCache:
    def __init__(self, default_ttl: int = EURECA_CACHE_TTL):
        self._store: dict[str, _CacheEntry] = {}
        self._ttl = default_ttl

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry and time.time() < entry.expires_at:
            return entry.value
        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._store[key] = _CacheEntry(
            value=value,
            expires_at=time.time() + (ttl or self._ttl),
        )

    def invalidate(self, prefix: str = "") -> None:
        if prefix:
            keys = [k for k in self._store if k.startswith(prefix)]
        else:
            keys = list(self._store.keys())
        for k in keys:
            del self._store[k]


_cache = _TTLCache()


# ---------------------------------------------------------------------------
# Session HTTP com retry automático
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_session = _make_session()


# ---------------------------------------------------------------------------
# Autenticação
# ---------------------------------------------------------------------------

@dataclass
class EurecaToken:
    access_token: str
    expires_at: float      # timestamp unix
    login: str = ""

    @property
    def is_valid(self) -> bool:
        return bool(self.access_token) and time.time() < self.expires_at - 60

    @property
    def header(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token}"}


_token_cache: dict[str, EurecaToken] = {}  # login → token


def authenticate(login: str, password: str) -> EurecaToken:
    """
    Autentica com credenciais do SIGAA e retorna um token JWT.

    Args:
        login:    Login SIGAA (ex: "123456789")
        password: Senha SIGAA

    Returns:
        EurecaToken com o JWT e validade.

    Raises:
        ValueError: Se as credenciais forem inválidas.
        requests.RequestException: Se a API estiver indisponível.
    """
    # Reutiliza token válido em cache
    cached = _token_cache.get(login)
    if cached and cached.is_valid:
        logger.debug("[eureca] reutilizando token para login=%s", login[:4] + "****")
        return cached

    logger.info("[eureca] autenticando login=%s****", login[:4])

    try:
        resp = _session.post(
            EURECA_AUTH_URL,
            json={"login": login, "senha": password},
            timeout=EURECA_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise requests.RequestException(
            f"Eureca auth indisponível: {exc}"
        ) from exc

    if resp.status_code == 401:
        raise ValueError("Credenciais SIGAA inválidas. Verifique login e senha.")
    if resp.status_code == 403:
        raise ValueError("Acesso negado pela API Eureca.")

    resp.raise_for_status()
    data = resp.json()

    # A API retorna { "token": "...", "expiresIn": 86400 } ou similar
    token_str  = data.get("token") or data.get("access_token") or data.get("accessToken", "")
    expires_in = data.get("expiresIn") or data.get("expires_in") or 86400  # default 24h

    if not token_str:
        raise ValueError(f"Token não encontrado na resposta: {list(data.keys())}")

    token = EurecaToken(
        access_token=token_str,
        expires_at=time.time() + int(expires_in),
        login=login,
    )
    _token_cache[login] = token
    logger.info("[eureca] token obtido, válido por %dh", expires_in // 3600)
    return token


def get_system_token() -> EurecaToken | None:
    """
    Retorna o token do sistema (configurado via EURECA_LOGIN/EURECA_PASSWD).
    Retorna None se as credenciais não estiverem configuradas.
    """
    if not EURECA_LOGIN or not EURECA_PASSWD:
        return None
    return authenticate(EURECA_LOGIN, EURECA_PASSWD)


# ---------------------------------------------------------------------------
# Chamadas à API
# ---------------------------------------------------------------------------

def _get(endpoint: str, token: EurecaToken, params: dict | None = None) -> Any:
    """
    Executa GET autenticado na API Eureca.
    Retorna o JSON parseado ou levanta exceção.
    """
    url = f"{EURECA_API_URL}/{endpoint.lstrip('/')}"
    logger.debug("[eureca] GET %s params=%s", url, params)

    try:
        resp = _session.get(
            url,
            headers={**token.header, "Content-Type": "application/json"},
            params=params or {},
            timeout=EURECA_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise requests.RequestException(f"Erro ao chamar Eureca: {exc}") from exc

    if resp.status_code == 401:
        # Token expirado — invalida cache e levanta para o chamador reautenticar
        _token_cache.pop(token.login, None)
        raise ValueError("Token Eureca expirado. Reautentique.")
    if resp.status_code == 404:
        return None   # recurso não encontrado — retorna None em vez de exceção
    if resp.status_code == 403:
        raise PermissionError("Sem permissão para acessar este recurso.")

    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Endpoints públicos (não requerem token de aluno)
# ---------------------------------------------------------------------------

def get_curriculo_componentes(curriculo_id: str, token: EurecaToken) -> list[dict]:
    """
    Retorna todas as disciplinas (componentes) de um currículo.

    Args:
        curriculo_id: Código do currículo (ex: "14102100" para CC/UFCG)
        token:        Token de autenticação

    Returns:
        Lista de componentes: [{codigo, nome, creditos, tipo, prerequisitos, ...}]
    """
    cache_key = f"curriculo:{curriculo_id}:componentes"
    cached = _cache.get(cache_key)
    if cached is not None:
        logger.debug("[eureca] cache hit: %s", cache_key)
        return cached

    data = _get(f"curriculo/{curriculo_id}/componentes", token)
    result = data if isinstance(data, list) else data.get("componentes", []) if data else []

    _cache.set(cache_key, result)
    logger.info("[eureca] curriculo %s: %d componentes", curriculo_id, len(result))
    return result


def get_componente_detalhes(curriculo_id: str, codigo: str, token: EurecaToken) -> dict | None:
    """
    Retorna detalhes de um componente curricular, incluindo pré-requisitos.

    Args:
        curriculo_id: Código do currículo
        codigo:       Código da disciplina (ex: "COMP3501")
        token:        Token de autenticação

    Returns:
        Dict com: codigo, nome, ementa, creditos, cargaHoraria, prerequisitos,
                  corequisitos, equivalencias, tipo
        None se não encontrado.
    """
    cache_key = f"curriculo:{curriculo_id}:componente:{codigo}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    data = _get(f"curriculo/{curriculo_id}/componente/{codigo}", token)
    if data:
        _cache.set(cache_key, data)
        logger.info("[eureca] componente %s carregado (prereqs=%d)",
                    codigo, len(data.get("prerequisitos", [])))
    return data


def get_turmas(
    periodo: str,
    token: EurecaToken,
    componente: str | None = None,
    curriculo_id: str | None = None,
) -> list[dict]:
    """
    Retorna as turmas ofertadas em um período, com horários e salas.

    Args:
        periodo:      Período letivo (ex: "2025.1")
        token:        Token de autenticação
        componente:   Filtra por código de disciplina (opcional)
        curriculo_id: Filtra por currículo (opcional)

    Returns:
        Lista de turmas: [{
            codigo, nome, turma, horarios: [{dia, hora, sala}],
            vagas, professor, local
        }]
    """
    cache_key = f"turmas:{periodo}:{componente or 'all'}:{curriculo_id or 'all'}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    params: dict = {"periodo": periodo}
    if componente:
        params["componente"] = componente
    if curriculo_id:
        params["curriculo"] = curriculo_id

    data = _get("turmas", token, params)
    result = data if isinstance(data, list) else data.get("turmas", []) if data else []

    # Cache curto para turmas (mudam durante a matrícula)
    _cache.set(cache_key, result, ttl=600)  # 10 min
    logger.info("[eureca] turmas %s %s: %d resultados", periodo, componente or "*", len(result))
    return result


def get_horarios_componente(
    componente: str,
    periodo: str,
    token: EurecaToken,
) -> list[dict]:
    """
    Atalho: retorna os horários de um componente específico num período.

    Returns:
        Lista de horários normalizados: [{turma, dia, hora_inicio, hora_fim, sala, professor}]
    """
    turmas = get_turmas(periodo=periodo, token=token, componente=componente)
    horarios = []

    for turma in turmas:
        turma_id = turma.get("turma") or turma.get("id", "")
        professor = turma.get("professor", {})
        prof_nome = professor.get("nome", "") if isinstance(professor, dict) else str(professor)

        for h in turma.get("horarios", []):
            dia  = h.get("dia", "")
            hora = h.get("hora", "") or h.get("horario", "")
            sala = h.get("sala", "") or h.get("local", "")

            # Normaliza hora para HH:MM-HH:MM se vier só como "08" (hora de início)
            hora_inicio, hora_fim = _parse_hora(hora)

            horarios.append({
                "turma":       turma_id,
                "dia":         _normalize_dia(dia),
                "hora_inicio": hora_inicio,
                "hora_fim":    hora_fim,
                "sala":        sala,
                "professor":   prof_nome,
                "vagas":       turma.get("vagas", 0),
            })

    return horarios


def get_prerequisitos(curriculo_id: str, codigo: str, token: EurecaToken) -> list[str]:
    """
    Retorna a lista de pré-requisitos de uma disciplina diretamente da API.

    Returns:
        Lista de códigos de pré-requisitos (ex: ["COMP2401", "COMP2201"])
        Lista vazia se sem pré-requisitos ou não encontrado.
    """
    cache_key = f"prereqs:{curriculo_id}:{codigo}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    detalhes = get_componente_detalhes(curriculo_id, codigo, token)
    if not detalhes:
        return []

    prereqs_raw = detalhes.get("prerequisitos") or detalhes.get("preRequisitos") or []

    # Normaliza: pode vir como lista de strings, lista de dicts, ou string separada por vírgula
    prereqs: list[str] = []
    for p in prereqs_raw:
        if isinstance(p, str):
            prereqs.append(p.strip())
        elif isinstance(p, dict):
            code = p.get("codigo") or p.get("code") or p.get("componente", "")
            if code:
                prereqs.append(str(code).strip())

    _cache.set(cache_key, prereqs)
    return prereqs


# ---------------------------------------------------------------------------
# Endpoints que requerem token do ALUNO
# ---------------------------------------------------------------------------

def get_historico_aluno(matricula: str, token: EurecaToken) -> list[dict]:
    """
    Retorna o histórico acadêmico do aluno autenticado.
    REQUER token do próprio aluno (não do sistema).

    Args:
        matricula: Número de matrícula do aluno
        token:     Token do PRÓPRIO aluno (obtido via authenticate())

    Returns:
        Lista de disciplinas cursadas: [{
            codigo, nome, periodo, nota, situacao, creditos
        }]

    Raises:
        PermissionError: Se tentar acessar histórico de outro aluno.
    """
    cache_key = f"historico:{matricula}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    data = _get(f"aluno/{matricula}/historico", token)
    result = data if isinstance(data, list) else data.get("historico", []) if data else []

    _cache.set(cache_key, result, ttl=300)  # 5 min — dados pessoais têm TTL curto
    logger.info("[eureca] histórico aluno %s: %d registros", matricula[:4] + "****", len(result))
    return result


def get_disciplinas_cursadas(matricula: str, token: EurecaToken) -> list[str]:
    """
    Retorna lista de CÓDIGOS de disciplinas em que o aluno foi aprovado.
    Útil para verificação automática de pré-requisitos.

    Returns:
        Lista de códigos aprovados (ex: ["COMP1001", "MAT1001", ...])
    """
    historico = get_historico_aluno(matricula, token)
    aprovados = [
        h.get("codigo", "")
        for h in historico
        if _is_aprovado(h.get("situacao", ""))
    ]
    return [c for c in aprovados if c]


def get_vinculo_aluno(matricula: str, token: EurecaToken) -> dict | None:
    """
    Retorna o vínculo ativo do aluno (curso, período atual, currículo, situação).

    Returns:
        Dict: {curso, curriculo, periodoAtual, situacao, ...} ou None
    """
    cache_key = f"vinculo:{matricula}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    data = _get(f"aluno/{matricula}/vinculo", token)
    if data:
        _cache.set(cache_key, data, ttl=3600)
    return data


# ---------------------------------------------------------------------------
# Helpers de normalização
# ---------------------------------------------------------------------------

def _parse_hora(hora_str: str) -> tuple[str, str]:
    """
    Normaliza string de horário para (inicio, fim).
    Formatos conhecidos: "08" → "08:00", "10:00", "08:00-10:00", "08h-10h"
    """
    if not hora_str:
        return "", ""

    hora_str = hora_str.strip().replace("h", ":00").replace("H", ":00")

    if "-" in hora_str:
        parts = hora_str.split("-", 1)
        inicio = _fmt_hora(parts[0].strip())
        fim    = _fmt_hora(parts[1].strip())
        return inicio, fim

    # Só hora de início — deduz fim por 2h (padrão UFCG)
    inicio = _fmt_hora(hora_str)
    try:
        h = int(hora_str.split(":")[0])
        fim = f"{h+2:02d}:00"
    except ValueError:
        fim = ""
    return inicio, fim


def _fmt_hora(s: str) -> str:
    """Formata '8' ou '08' para '08:00'."""
    s = s.strip()
    if ":" not in s:
        try:
            return f"{int(s):02d}:00"
        except ValueError:
            return s
    parts = s.split(":")
    return f"{int(parts[0]):02d}:{parts[1].zfill(2)}"


_DIAS_MAP = {
    "seg":    "segunda",
    "segunda": "segunda",
    "mon":    "segunda",
    "ter":    "terca",
    "terca":  "terca",
    "tue":    "terca",
    "qua":    "quarta",
    "quarta": "quarta",
    "wed":    "quarta",
    "qui":    "quinta",
    "quinta": "quinta",
    "thu":    "quinta",
    "sex":    "sexta",
    "sexta":  "sexta",
    "fri":    "sexta",
    "sab":    "sabado",
    "sabado": "sabado",
    "sat":    "sabado",
    "2":      "segunda",
    "3":      "terca",
    "4":      "quarta",
    "5":      "quinta",
    "6":      "sexta",
}


def _normalize_dia(dia: str) -> str:
    return _DIAS_MAP.get(dia.lower().strip(), dia.lower().strip())


def _is_aprovado(situacao: str) -> bool:
    """Retorna True se a situação indica aprovação."""
    s = situacao.lower()
    return any(kw in s for kw in ["aprovado", "aprovada", "apto", "concluido", "pass"])


# ---------------------------------------------------------------------------
# Cliente de alto nível (facade para uso nos agentes)
# ---------------------------------------------------------------------------

class EurecaClient:
    """
    Facade de alto nível para uso nos agentes.
    Encapsula autenticação e expõe métodos semânticos.
    """

    def __init__(self, login: str = "", password: str = ""):
        self._login    = login    or EURECA_LOGIN
        self._password = password or EURECA_PASSWD
        self._token: EurecaToken | None = None

    @property
    def token(self) -> EurecaToken:
        if not self._token or not self._token.is_valid:
            if not self._login:
                raise ValueError(
                    "Credenciais Eureca não configuradas. "
                    "Defina EURECA_LOGIN e EURECA_PASSWD no .env."
                )
            self._token = authenticate(self._login, self._password)
        return self._token

    def get_prerequisitos(self, codigo: str, curriculo_id: str = "14102100") -> list[str]:
        """Pré-requisitos de uma disciplina (dados reais da API)."""
        return get_prerequisitos(curriculo_id, codigo, self.token)

    def get_horarios(self, codigo: str, periodo: str) -> list[dict]:
        """Horários de uma disciplina num período (dados reais da API)."""
        return get_horarios_componente(codigo, periodo, self.token)

    def verificar_conflito(self, codigos: list[str], periodo: str) -> list[dict]:
        """
        Verifica conflitos de horário entre uma lista de disciplinas.
        Retorna lista de conflitos [{curso_a, curso_b, dia, hora, descricao}].
        """
        # Busca horários de todas as disciplinas
        horarios_por_codigo: dict[str, list[dict]] = {}
        for codigo in codigos:
            try:
                hs = self.get_horarios(codigo, periodo)
                if hs:
                    horarios_por_codigo[codigo] = hs
            except Exception as exc:
                logger.warning("[eureca] falha ao buscar horários de %s: %s", codigo, exc)

        conflicts = []
        codigos_com_horario = list(horarios_por_codigo.keys())

        for i in range(len(codigos_com_horario)):
            for j in range(i + 1, len(codigos_com_horaria := codigos_com_horario)):
                a, b = codigos_com_horaria[i], codigos_com_horaria[j]
                for ha in horarios_por_codigo[a]:
                    for hb in horarios_por_codigo[b]:
                        if ha["dia"] == hb["dia"] and _horarios_se_sobrepõem(ha, hb):
                            conflicts.append({
                                "curso_a":    a,
                                "curso_b":    b,
                                "dia":        ha["dia"],
                                "hora_a":     f"{ha['hora_inicio']}–{ha['hora_fim']}",
                                "hora_b":     f"{hb['hora_inicio']}–{hb['hora_fim']}",
                                "sala_a":     ha.get("sala", ""),
                                "sala_b":     hb.get("sala", ""),
                                "descricao":  (
                                    f"{a} ({ha['hora_inicio']}–{ha['hora_fim']}) × "
                                    f"{b} ({hb['hora_inicio']}–{hb['hora_fim']}) "
                                    f"na {ha['dia']}-feira"
                                ),
                            })
        return conflicts

    def historico_aluno(self, matricula: str) -> list[dict]:
        """Histórico acadêmico — requer token do próprio aluno."""
        return get_historico_aluno(matricula, self.token)

    def disciplinas_aprovadas(self, matricula: str) -> list[str]:
        """Códigos de disciplinas aprovadas pelo aluno."""
        return get_disciplinas_cursadas(matricula, self.token)

    def vinculo_aluno(self, matricula: str) -> dict | None:
        """Vínculo ativo do aluno."""
        return get_vinculo_aluno(matricula, self.token)


def _horarios_se_sobrepõem(ha: dict, hb: dict) -> bool:
    """Verifica sobreposição de horário entre dois slots."""
    try:
        def to_min(t: str) -> int:
            h, m = t.split(":") if ":" in t else (t, "0")
            return int(h) * 60 + int(m)

        s_a, e_a = to_min(ha["hora_inicio"]), to_min(ha["hora_fim"])
        s_b, e_b = to_min(hb["hora_inicio"]), to_min(hb["hora_fim"])
        return s_a < e_b and s_b < e_a
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI para testes rápidos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json
    import getpass
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Uso: python -m src.integrations.eureca_client <comando> [args]")
        print("Comandos:")
        print("  prereqs <curriculo> <codigo>")
        print("  horarios <codigo> <periodo>")
        print("  conflito <periodo> <cod1> <cod2> ...")
        sys.exit(0)

    login    = input("Login SIGAA: ")
    password = getpass.getpass("Senha SIGAA: ")
    client   = EurecaClient(login, password)

    cmd = sys.argv[1]

    if cmd == "prereqs" and len(sys.argv) >= 4:
        result = client.get_prerequisitos(sys.argv[3], sys.argv[2])
        print(f"Pré-requisitos de {sys.argv[3]}: {result}")

    elif cmd == "horarios" and len(sys.argv) >= 4:
        result = client.get_horarios(sys.argv[2], sys.argv[3])
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif cmd == "conflito" and len(sys.argv) >= 5:
        codigos = sys.argv[3:]
        result  = client.verificar_conflito(codigos, sys.argv[2])
        if result:
            print(f"⚠️  {len(result)} conflito(s) detectado(s):")
            for c in result:
                print(f"  {c['descricao']}")
        else:
            print("✅ Sem conflitos de horário.")

    else:
        print("Argumentos insuficientes.")