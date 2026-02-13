# 🗺️ MAPA DEL FLUJO MAESTRO: ENJAMBRE AIGOTAJOB & ESY BISNE

## 🏗️ ARQUITECTURA DEL SISTEMA (Activepieces + AI)

### 1. CAPH-TRAP (Fuentes de Leads)
*   **A: Portales de Empleo** (Moovijob, LinkedIn) -> [Filtro: Empresas Contratando]
*   **B: Redes Hyper-Locales** (Hoplr, Facebook Groups) -> [Filtro: Nuevos Negocios / Webs de Cartón]
*   **C: Portales Inmobiliarios** (atHome.lu, Vivi.lu) -> [Filtro: Alquileres Particulares < 1500€]

### 2. EL CEREBRO (Activepieces Logic)
*   **Nodo Clasificador AI:** Decide la ruta del lead (AiGotAJob o Esy Bisne).
*   **Nodo Enriquecedor (Lusha API):** Encuentra el email del CEO o Manager.
*   **Nodo Financiero (Stripe):** Valida si el usuario de alquileres tiene la suscripción activa.

### 3. ACCIÓN & ENTREGA (Output)
*   **AiGotAJob:** Envío de Landing de Candidato a Reclutador.
*   **Esy Bisne (Digital):** Creación de Propuesta en Google Doc -> Notificación a Artesano.
*   **Esy Bisne (Alquileres):** WhatsApp/Email VIP con link directo a la oferta filtrada.

---

## 💾 REGISTRO CENTRAL (Google Sheets / Supabase)
*   `TABLE_LEADS`: Historial de todo lo cazado.
*   `TABLE_VIP_USERS`: Clientes que pagan suscripción de alquileres.
*   `TABLE_ARTESANOS`: Nuestra base de datos de profesionales para ejecutar trabajos.

---
*Documento de Planificación Estratégica - Comandante Yusmel*
