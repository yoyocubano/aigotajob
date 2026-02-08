# 🦅 AiGotAJob: Agente Autónomo de Empleo en Luxemburgo

## 📖 ¿Qué es AiGotAJob?
**AiGotAJob** es un enjambre de bots especializados en el mercado laboral de Luxemburgo. Su misión es vigilar las 24 horas del día todas las plataformas de empleo del Gran Ducado (ADEM, Moovijob, Jobs.lu, LinkedIn y Facebook) para identificar oportunidades de trabajo y conectar automáticamente a candidatos con reclutadores.

El producto estrella que impulsa este enjambre es la **Modern CV Landing Page**: una presencia web premium que sustituye al CV tradicional de PDF, haciendo que el candidato sea irresistible para las empresas luxemburguesas.

---

## 🛰️ Cómo Funciona (Tecnología)

### 1. Vigilancia Omnicanal (The Snipers)
El bot utiliza **Selenium WebDriver** para navegar de forma indetectable (Protocolo Fantasma) por:
- **ADEM (jobboard.adem.lu):** Acceso directo a las ofertas del servicio público de empleo.
- **Moovijob & Jobs.lu:** Escaneo de ofertas premium y eventos de reclutamiento.
- **Redes Sociales:** Monitoreo de grupos de Facebook y perfiles "Open to Work" en LinkedIn.

### 2. Cerebro AI (Decision Engine)
Cada oferta detectada es analizada mediante modelos de lenguaje para asegurar que el sector coincide con nuestros objetivos. El bot extrae:
- Nombre de la empresa / Reclutador.
- Requisitos del puesto.
- Idioma de la oferta (Francés, Inglés, Alemán).

### 3. Respuesta Automática (Neural Outreach)
Utiliza el **Neural Response Engine** para enviar mensajes personalizados que ofrecen el CV Moderno (50€) como la solución definitiva a la "invisibilidad" laboral.

---

## 🛠️ Stack Tecnológico
- **Core:** Python 3.9+
- **Browser Automation:** Selenium (Hardened Mode)
- **Persistencia:** SQL Hybrid (Sincronización con EsyBisne DB)
- **Seguridad:** Gestión de credenciales cifradas vía `AIGOTAJOB_VAULT.json`

---

## 🚀 Futuro y Escalabilidad
- **Módulos ADEM Pro:** Automatización de postulaciones directas con el perfil oficial.
- **AI Interview Prep:** Integración de un asistente de IA que prepara al candidato para la entrevista basada en la oferta detectada.
- **Gestión Multi-Perfil:** Posibilidad de que agencias de empleo usen AiGotAJob para gestionar cientos de candidatos simultáneamente.

---

### Proyecto Hermano de EsyBisne App
*Código impulsado por Antigravity Core - 2026-02-08*
