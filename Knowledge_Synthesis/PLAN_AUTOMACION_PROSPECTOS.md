# 🦅 PLAN ESTRATÉGICO: Automación de Prospectos B2B
**Proyecto:** AiGotAJob (Enjambre de Prospección)
**Herramientas:** Lusha + n8n + Maxun

---

## 🎯 Objetivo
Escalar la captación de clientes B2B (Reclutadores, CEOs y Managers en Luxemburgo) mediante la extracción masiva de datos y el enriquecimiento automático de contactos para envío de correos directos.

## 🏗️ Componentes del Sistema

### 1. Extracción (The Hunter - Maxun)
*   **Función:** Scrapear perfiles de LinkedIn o directorios de empresas en Luxemburgo.
*   **Output:** Lista de nombres, cargos y empresas en formato JSON o CSV.

### 2. Enriquecimiento (The Oracle - Lusha API)
*   **Función:** Usar la API de Lusha para convertir perfiles sociales en datos de contacto directos.
*   **Datos Clave:** Email corporativo verificado y teléfono directo.
*   **Costo:** Gestión de créditos vía API.

### 3. Orquestación (The Brain - n8n Local)
*   **Función:** Unir todas las piezas en un flujo de trabajo automático.
*   **Localización:** Corriendo en `http://localhost:5678`.
*   **Nodos Clave:** HTTP Request (Lusha API), Gmail/NodeMailer (Outreach), Supabase/Google Sheets (Base de Datos).

---

## 🔄 Workflow Propuesto (El Flujo del Éxito)

1.  **Trigger:** Maxun detecta nuevos perfiles "Open to Work" o "Hiring" en Luxemburgo.
2.  **Filtro:** n8n valida si el contacto ya existe en `LUXJOB_VAULT.json` o Supabase.
3.  **Enriquecimiento:** Si es nuevo, n8n llama a la API de Lusha y recupera el email.
4.  **Acción:** n8n envía un correo personalizado con el pitch de "Modern CV Landing Page" (AiGotAJob).
5.  **Registro:** Se guarda el log en la base de datos para seguimiento.

---

## 📍 Próximos Pasos (To-Do)
- [ ] Configurar el primer robot en **Maxun** para LinkedIn (Luxemburgo).
- [ ] Validar la API Key de **Lusha** (usar créditos gratuitos iniciales).
- [ ] Diseñar el Workflow en **n8n local** (Nodos: HTTP Request -> IF -> Send Email).

---
*Documento generado por Antigravity para el Comandante Yoyocubano.*
*Fecha: 2026-02-12*
