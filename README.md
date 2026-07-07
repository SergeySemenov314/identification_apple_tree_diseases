# Определение болезней листьев яблони

Веб-приложение: пользователь загружает фото листа яблони (файл, камера или галерея
тестовых фото) → модель определяет болезни и возвращает их **строкой** с вероятностями.
Фото обратно не возвращается.

Доступно по пути **`/apple-tree-diseases`** через главный nginx
(`https://5phm.l.time4vps.cloud/apple-tree-diseases`).

## Модель

- **Задача:** мультилейбл-классификация (один лист → несколько болезней).
- **Архитектура:** EfficientNet-B0 (ImageNet) + голова `Dropout(0.3) + Linear(6)`, Sigmoid + BCE.
- **Классы (6):** `complex`, `frog_eye_leaf_spot`, `healthy`, `powdery_mildew`, `rust`, `scab`.
- **Веса:** `models/best_model_efficientnet_b0_v2.pt` (Plant Pathology 2021 — FGVC8).
  В чекпойнте лежат `idx2label`, `thresholds` (пороги по классам) и `cfg` (`IMG_SIZE=256`).
- **Инференс:** Resize 256×256 → ImageNet-нормализация → TTA (оригинал + h-flip + v-flip),
  усреднение вероятностей. Класс попадает в диагноз, если его вероятность ≥ порога класса;
  если ни один порог не пройден — берётся класс с максимальной вероятностью.

## Структура

```
identification_apple_tree_diseases/
├── docker-compose.yml         # 3 сервиса в сети ollama-net
├── inference_service/         # FastAPI + PyTorch (порт 8001, внутренний)
├── backend/                   # Node/Express: галерея + прокси на инференс (порт 3002)
├── frontend/                  # React (nginx :80), путь /apple-tree-diseases
├── models/                    # .pt веса (монтируются в inference как /app/model, ro)
└── test_photos/               # галерея тестовых фото (монтируется в backend, ro)
```

Сервисы (имена контейнеров): `apple-inference`, `apple-backend`, `apple-frontend`.

## Запуск на VPS

Сеть `ollama-net` уже существует. Из папки проекта:

```bash
cd identification_apple_tree_diseases
docker compose up -d --build
```

Первая сборка `apple-inference` долгая — тянет torch (CPU). Проверить статус:

```bash
docker compose ps
docker compose logs -f apple-inference   # дождаться "[startup] model=..."
```

### Подключение маршрута в главном nginx

Маршруты `/apple-tree-diseases/` и `/apple-tree-diseases/api/` уже добавлены в
`models_chat_react/nginx-ssl.conf`. Чтобы они заработали, пересоберите/перезапустите
контейнер главного nginx (тот, что отдаёт 443 из `models_chat_react`), например:

```bash
cd ../models_chat_react
docker compose up -d --build   # либо docker restart <имя nginx-контейнера>
```

## Проверка

- API инференса: `docker exec apple-inference curl -s localhost:8001/health`
- Бэкенд: `docker exec apple-backend wget -qO- localhost:3002/health`
- Интерфейс: `https://5phm.l.time4vps.cloud/apple-tree-diseases`

## Формат ответа `/api/detect`

```json
{
  "labels": "scab rust",
  "is_healthy": false,
  "diseases": [
    { "label": "scab", "name_ru": "Парша", "description": "...", "probability": 0.87 }
  ],
  "predictions": [
    { "label": "scab", "name_ru": "Парша", "probability": 0.87, "threshold": 0.45, "detected": true }
  ]
}
```
