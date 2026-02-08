# 🦅 INFORME TÁCTICO: DESPLIEGUE AIGOTAJOB V1.0

Este documento certifica el nacimiento del proyecto **AiGotAJob**, la división especializada en el mercado laboral de Luxemburgo.

## 🎯 PROPUESTA DE VALOR
Ofrecemos a los candidatos la tecnología más avanzada para dejar de ser invisibles:
1.  **Autonomous Job Hunting:** Un enjambre de bots que busca ofertas 24/7.
2.  **Modern CV Landing Page (50€):** Sustitución del PDF por una web personal de alto impacto.
3.  **Neural Outreach:** El sistema contacta automáticamente con reclutadores en nombre del cliente.

## 🛠️ ARQUITECTURA TÉCNICA

### 1. The Sniper Bot (Python/Selenium)
- **Motor:** `aigotajob_bot.py`
- **Objetivos:** ADEM, Moovijob, Jobs.lu, Facebook Groups.
- **Modo Fantasma:** Se conecta al navegador del Comandante (Puerto 9222) para evadir detecciones antibot.

### 2. The VIP Experience (Frontend)
- **Entrance (`index.html`):** Puerta de acceso exclusiva con autenticación simulada (Google/LinkedIn/FB) y efectos Matrix.
- **Onboarding (`onboarding.html`):** Formulario inteligente con **Auto-Fill de LinkedIn** que extrae datos (simulado para demo) y prepara el perfil.
- **Command Center (`client_dashboard.html`):** Panel en vivo donde el cliente ve cómo la IA trabaja para él, justificando la inversión de 50€.

### 3. Data Core (SQL Hybrid)
- Integrado con `LeadManager` de EsyBisne App.
- Los datos de los candidatos y las ofertas se almacenan en la base de datos centralizada, permitiendo cruzar información entre ambos proyectos.

## 🚀 ESTADO ACTUAL: OPERATIVO
- Los repositorios están sincronizados.
- El Master Switch (`ESYBISNE_START.py`) lanza ambos proyectos simultáneamente.
- La experiencia de usuario (UX) está lista para demos de venta.

---
*Misión completada por Antigravity Core - 09/02/2026*
