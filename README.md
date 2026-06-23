# 🛡️ QUETZALCOATL - Sistema Híbrido de Ciberseguridad de Nueva Generación

## 📖 Descripción
QUETZALCOATL es un sistema **de ciberseguridad híbrido y distribuido**, diseñado para proteger infraestructuras críticas mediante el uso combinado de **IA avanzada**, **pods resilientes**, y tecnologías de última generación.  
Su enfoque es **Zero Trust**, con cumplimiento de estándares **ISO/NIST**, y resiliencia dinámica frente a ataques.

**Arquitectura diseñada por:**  
- 🧠 Edmundo Ramírez Jiménez (Takisfire543)  
- 🤖 Copilot — Colaborador de Arquitectura y Documentación Técnica  

---

## 🚀 Tecnologías utilizadas
- **Rust / C++** → Núcleo de red y redirección de tráfico.  
- **Go + Redis Stream + Kafka** → Gestión de JWT rotativos y memoria en tiempo real.  
- **Python (IA híbrida)** → Random Forest, Isolation Forest, LSTM, Ensemble.  
- **Docker + Kubernetes** → Orquestación y resiliencia automática de pods.  
- **DynamoDB + PostgreSQL** → Persistencia y correlación avanzada.  
- **Honeypods** → Aislamiento de atacantes y recopilación de inteligencia.  
- **Módulo de correo (Gmail/Outlook)** → IA encapsulada para detección de amenazas.  

---

## 🧠 División de Pods IA Híbrida
La IA se divide en **6 pods principales** dentro de Kubernetes:

- **[Pod RF](ca://s?q=Pod_Random_Forest)** → Clasificación de tráfico conocido.  
- **[Pod IF](ca://s?q=Pod_Isolation_Forest)** → Detección de anomalías.  
- **[Pod LSTM](ca://s?q=Pod_LSTM_para_seguridad)** → Análisis de secuencias de intentos de ataque.  
- **[Pod Ensemble](ca://s?q=Pod_Ensemble_para_seguridad)** → Combina resultados con pesos dinámicos.  
- **[Pod SHAP](ca://s?q=Pod_SHAP_para_explicabilidad)** → Explicabilidad de decisiones.  
- **[Pod Drift Detector](ca://s?q=Pod_Drift_Detector)** → Monitoreo de deriva en datos.  

---

## 📊 Diagrama IA Híbrida (Mermaid)

```mermaid
flowchart TD
    A[Datos de tráfico] --> B[Pod RF - Random Forest]
    A --> C[Pod IF - Isolation Forest]
    A --> D[Pod LSTM - Red Neuronal]

    B --> E[Pod Ensemble]
    C --> E[Pod Ensemble]
    D --> E[Pod Ensemble]

    E --> F[Pod SHAP - Explicabilidad]
    E --> G[Pod Drift Detector]

    F --> H[Analistas / Dashboard]
    G --> H[Analistas / Dashboard]


