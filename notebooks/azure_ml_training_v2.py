"""
Azure ML Training Script v2.0
Autor: Edmundo Ramírez Jiménez
Proyecto: HIBRID TECHNOLOGIES Cibersecurity
Festival: Habilidades IA (Junio 8-12, 2026)

Este script contiene el flujo de entrenamiento, evaluación, explicación (SHAP), detección de drift y registro
en MLflow y Azure ML descrito por el autor.
"""

# 1. Setup - Importar Librerías Mejorado

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, Environment, BuildContext, ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Endpoint, OnlineEndpoint, Model, ProbeSettings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib
import json
import logging
import mlflow
import shap
from datetime import datetime
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from alibi_detect.cd import MMDDrift
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("✅ Librerías importadas correctamente (v2.0)")

# 2. Conectar a Azure ML con MLflow
subscription_id = "YOUR_SUBSCRIPTION_ID"
resource_group = "cyber-sec-rg"
workspace_name = "cyber-sec-ml"

# Configurar MLflow
mlflow.set_tracking_uri(f"azureml://{subscription_id}.{resource_group}.{workspace_name}")
mlflow.set_experiment("cyber-sec-hybrid-ia")

credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

print(f"✅ Conectado a workspace: {workspace_name}")

# 3. Generar Dataset Realista Mejorado
np.random.seed(42)

def generate_cyber_data(n_samples=10000, n_features=50):
    """Genera datos sintéticos realistas de ciberseguridad"""

    # Tráfico normal (distribución centrada)
    X_normal = np.random.normal(0, 1, (n_samples // 2, n_features))
    y_normal = np.zeros(n_samples // 2)

    # Tráfico malicioso (con diferentes patrones)
    X_attack1 = np.random.normal(3, 2, (n_samples // 4, n_features))
    X_attack2 = np.random.normal(-2, 1.5, (n_samples // 8, n_features))
    X_attack3 = np.random.uniform(-5, 5, (n_samples // 8, n_features))  # Patrón ruidoso

    X_attack = np.vstack([X_attack1, X_attack2, X_attack3])
    y_attack = np.ones(X_attack.shape[0])

    # Combinar
    X = np.vstack([X_normal, X_attack])
    y = np.concatenate([y_normal, y_attack])

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    return X, y

# Generar datos
X, y = generate_cyber_data(n_samples=10000, n_features=50)

# Split mejorado (con estratificación)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Dataset generado con estratificación")
print(f"   - Training: {X_train.shape} (Malicioso: {y_train.sum()/len(y_train):.2%})")
print(f"   - Testing: {X_test.shape} (Malicioso: {y_test.sum()/len(y_test):.2%})")

# 4. Preprocesamiento y Feature Engineering
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature augmentation (nuevas características)
X_train_aug = np.hstack([
    X_train_scaled,
    np.mean(X_train_scaled, axis=1, keepdims=True),  # Media por muestra
    np.std(X_train_scaled, axis=1, keepdims=True),   # Std por muestra
    np.max(X_train_scaled, axis=1, keepdims=True),   # Max por muestra
    np.min(X_train_scaled, axis=1, keepdims=True),   # Min por muestra
])

X_test_aug = np.hstack([
    X_test_scaled,
    np.mean(X_test_scaled, axis=1, keepdims=True),
    np.std(X_test_scaled, axis=1, keepdims=True),
    np.max(X_test_scaled, axis=1, keepdims=True),
    np.min(X_test_scaled, axis=1, keepdims=True),
])

print(f"✅ Feature augmentation: {X_train_aug.shape[1]} características")

# 5. Entrenamiento con MLflow Tracking
with mlflow.start_run() as run:
    logger.info(f"🚀 Iniciando run MLflow: {run.info.run_id}")

    # 5.1 Random Forest
    logger.info("🌲 Entrenando Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,  # Aumentado
        max_depth=30,      # Aumentado
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_aug, y_train)

    # Métricas RF
    rf_pred = rf_model.predict(X_test_aug)
    rf_proba = rf_model.predict_proba(X_test_aug)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)
    logger.info(f"   ✅ ROC-AUC RF: {rf_auc:.4f}")

    # 5.2 Isolation Forest
    logger.info("🔍 Entrenando Isolation Forest...")
    if_model = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    if_model.fit(X_train_aug)
    if_anomaly = if_model.predict(X_test_aug)
    if_scores = np.where(if_anomaly == -1, 1, 0)
    logger.info(f"   ✅ Anomalías detectadas: {if_scores.sum()}")

    # 5.3 LSTM mejorado
    logger.info("🧠 Entrenando LSTM...")

    class ThreatLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
            super(ThreatLSTM, self).__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True  # ¡Nuevo! LSTM bidireccional
            )
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 por bidireccional
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            lstm_out, (h_n, c_n) = self.lstm(x)
            out = self.dropout(lstm_out[:, -1, :])
            out = self.relu(self.fc1(out))
            out = self.relu(self.fc2(out))
            out = self.sigmoid(self.fc3(out))
            return out

    # Configuración LSTM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_train_aug.shape[1]
    hidden_size = 128
    num_layers = 3
    num_epochs = 30
    batch_size = 64
    learning_rate = 0.001

    lstm_model = ThreatLSTM(input_size, hidden_size, num_layers, dropout=0.3).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Preparar datos
    X_train_tensor = torch.FloatTensor(X_train_aug).unsqueeze(1).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test_aug).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Entrenar LSTM
    train_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        lstm_model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Evaluar LSTM
    lstm_model.eval()
    with torch.no_grad():
        lstm_pred = lstm_model(X_test_tensor).cpu().numpy().flatten()
        lstm_loss = criterion(torch.FloatTensor(lstm_pred).to(device), y_test_tensor)
        lstm_auc = roc_auc_score(y_test, lstm_pred)
        logger.info(f"   ✅ LSTM Test Loss: {lstm_loss.item():.4f}, AUC: {lstm_auc:.4f}")

    # 6. Ensemble con pesos dinámicos
    aucs = {
        'random_forest': rf_auc,
        'isolation_forest': 0.75,  # Aproximado
        'lstm': lstm_auc
    }

    # Pesos proporcionales al AUC
    total_auc = sum(aucs.values())
    weights = {k: v / total_auc for k, v in aucs.items()}

    logger.info(f"📊 Pesos del ensemble: {weights}")

    # Predicción ensemble
    ensemble_score = (
        weights['random_forest'] * rf_proba +
        weights['isolation_forest'] * if_scores +
        weights['lstm'] * lstm_pred
    )

    y_pred_ensemble = (ensemble_score >= 0.5).astype(int)
    ensemble_auc = roc_auc_score(y_test, ensemble_score)

    # Métricas
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_ensemble, average='binary'
    )
    cm = confusion_matrix(y_test, y_pred_ensemble)

    logger.info(f"\n🎯 ENSEMBLE FINAL")
    logger.info(f"   ROC-AUC: {ensemble_auc:.4f}")
    logger.info(f"   Precision: {precision:.4f}")
    logger.info(f"   Recall: {recall:.4f}")
    logger.info(f"   F1-Score: {f1:.4f}")

    # 7. SHAP para explicabilidad
    logger.info("🧠 Generando explicaciones SHAP...")

    # Random Forest SHAP
    explainer_rf = shap.TreeExplainer(rf_model)
    shap_values_rf = explainer_rf.shap_values(X_test_aug[:100])  # Muestra para performance

    # Guardar SHAP summary plot
    os.makedirs('models', exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_rf[1], X_test_aug[:100], show=False)
    plt.savefig('models/shap_summary.png', bbox_inches='tight')
    plt.close()
    logger.info("   ✅ SHAP guardado")

    # 8. Drift Detection
    logger.info("📊 Configurando detección de drift...")
    drift_detector = MMDDrift(
        X_train_aug[:1000],
        p_val=0.05,
        backend='tensorflow'
    )

    # Test drift
    drift_scores = []
    for i in range(0, len(X_test_aug), 100):
        batch = X_test_aug[i:i+100]
        if len(batch) > 0:
            is_drift, p_val, _ = drift_detector.predict(batch)
            drift_scores.append(p_val if is_drift else 0)

    mean_drift = np.mean(drift_scores) if drift_scores else 0
    logger.info(f"   ✅ Drift score promedio: {mean_drift:.4f}")

    # 9. Logging en MLflow
    mlflow.log_params({
        'n_samples': len(X),
        'n_features': X.shape[1],
        'augmented_features': X_train_aug.shape[1],
        'rf_estimators': 200,
        'rf_max_depth': 30,
        'lstm_hidden_size': hidden_size,
        'lstm_layers': num_layers,
        'lstm_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'dropout': 0.3,
        'ensemble_weights': str(weights)
    })

    # Guardar métricas
    mlflow.log_metrics({
        'rf_auc': rf_auc,
        'lstm_auc': lstm_auc,
        'ensemble_auc': ensemble_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'drift_score': mean_drift,
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1])
    })

    # Guardar artifacts
    joblib.dump(rf_model, 'models/random_forest.pkl')
    joblib.dump(if_model, 'models/isolation_forest.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    torch.save(lstm_model.state_dict(), 'models/lstm_model.pth')

    # Guardar weights y config
    config = {
        'weights': weights,
        'input_shape': X_train_aug.shape[1],
        'aucs': aucs,
        'threshold': 0.5,
        'timestamp': datetime.now().isoformat(),
        'version': '2.0'
    }
    with open('models/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Registrar en MLflow
    mlflow.log_artifacts('models')

    # 10. Registrar en Azure ML
    model = Model(
        name="cyber-sec-ensemble-v2",
        path="./models",
        description="Ensemble mejorado para detección de ciberataques (RF + IF + LSTM + SHAP + Drift)",
        type=AssetTypes.CUSTOM_MODEL,
        tags={
            "project": "HybridTechnologies",
            "author": "Edmundo Ramirez Jimenez",
            "version": "2.0",
            "ensemble_auc": str(ensemble_auc),
            "f1_score": str(f1),
            "features": str(X_train_aug.shape[1])
        }
    )

    ml_client.models.create_or_update(model)
    logger.info(f"✅ Modelo registrado en Azure ML: {model.name}")

    # 11. Visualización de resultados
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Matriz de confusión
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')

    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, ensemble_score)
    axes[0, 1].plot(fpr, tpr, label=f'Ensemble (AUC = {ensemble_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()

    # Distribución de scores
    axes[1, 0].hist(ensemble_score[y_test == 0], alpha=0.5, label='Normal', bins=30)
    axes[1, 0].hist(ensemble_score[y_test == 1], alpha=0.5, label='Attack', bins=30)
    axes[1, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
    axes[1, 0].set_xlabel('Ensemble Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].legend()

    # Feature Importance (top 20)
    feature_names = [f'f{i}' for i in range(X_train_aug.shape[1])]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-20:]
    axes[1, 1].barh(range(20), importances[indices])
    axes[1, 1].set_yticks(range(20))
    axes[1, 1].set_yticklabels([feature_names[i] for i in indices])
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Top 20 Features')

    plt.tight_layout()
    plt.savefig('models/ensemble_analysis.png')
    plt.close()

    print("\n✅ Análisis completo guardado en: models/ensemble_analysis.png")

    # 12. Resumen Final
    summary = f""" \n╔════════════════════════════════════════════════════════════╗\n║         🛡️ CyberSec Hybrid IA v2.0 - TRAINING COMPLETE  ║\n╚════════════════════════════════════════════════════════════╝\n\n📊 DATASET • Total samples: {len(X)} • Features originales: {X.shape[1]} • Features aumentadas: {X_train_aug.shape[1]} • Split: 80/20 con estratificación\n\n🤖 MODELOS\n\nRandom Forest (200 árboles, depth=30) • ROC-AUC: {rf_auc:.4f}\n\nIsolation Forest (contamination=0.1) • Anomalías: {if_scores.sum()}\n\nLSTM Bidireccional (3 capas, hidden=128) • ROC-AUC: {lstm_auc:.4f}\n\n⚖️ ENSEMBLE (Pesos dinámicos por AUC) • RF: {weights['random_forest']:.3f} • IF: {weights['isolation_forest']:.3f} • LSTM: {weights['lstm']:.3f}\n\n🎯 MÉTRICAS FINALES • ROC-AUC: {ensemble_auc:.4f} • Precision: {precision:.4f} • Recall: {recall:.4f} • F1-Score: {f1:.4f} • Accuracy: {(y_test == y_pred_ensemble).mean():.4f} • Drift Score: {mean_drift:.4f}\n\n📁 ARTIFACTOS GUARDADOS ✅ Modelos: rf, if, lstm, scaler ✅ Configuración: config.json ✅ Análisis: ensemble_analysis.png ✅ SHAP: shap_summary.png ✅ MLflow run: {run.info.run_id}\n\n🚀 REGISTRADO EN AZURE ML • Modelo: cyber-sec-ensemble-v2 • Workspace: {workspace_name}✅ LISTO PARA DEPLOYMENT COMO Idpoint?"""
    
    logger.info(summary)
