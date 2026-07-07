import React, { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';

const API = process.env.REACT_APP_API_URL || '/apple-tree-diseases';

const ML_DESCRIPTION = [
  {
    title: 'О задаче',
    text: (
      <>
        Модель создана для соревнования{' '}
        <a
          href="https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8"
          target="_blank"
          rel="noopener noreferrer"
        >
          Plant Pathology 2021
        </a>{' '}
        на Kaggle (организовано в рамках конференции CVPR; соревнование завершено). Практическая проблема: болезни листьев — одна из главных угроз урожаю яблонь, а их диагностика в садах во многом основана на ручном осмотре, который является трудоемким и дорогостоящим. Цель — автоматически определять заболевание по фотографии листа, чтобы ускорить и удешевить мониторинг сада.<br /><br />
        Сложность задачи в том, что один и тот же диагноз визуально сильно варьируется (разные сорта, возраст ткани, освещение, фон, ракурс), а на одном листе может присутствовать сразу несколько болезней.
      </>
    )
  },
  {
    title: 'Датасет',
    text: 'За основу взят публичный датасет соревнования: 23 000 фотографий листьев яблони, снятые в реальных садах при разных условиях (разное освещение, время суток, фон, стадии зрелости листа). Разметку выполняли эксперты-фитопатологи — то есть диагнозы проставлены специалистами по болезням растений. Один лист может иметь сразу несколько диагнозов. Классы: healthy (здоровый), scab (парша), rust (ржавчина), frog_eye_leaf_spot (бурая пятнистость), powdery_mildew (мучнистая роса) и complex (комплекс болезней). Оригинальные изображения крупные (~4000 px, ~16 ГБ), поэтому для обучения они предварительно уменьшены до 512 px — это устранило узкое место с декодированием при обучении.'
  },
  {
    title: 'Модель',
    text: (
      <>
        Были обучены и сопоставлены три архитектуры: EfficientNet-B0 (лёгкий CNN), ConvNeXt-Tiny (модернизированный CNN) и Swin-Tiny (трансформер). По итогам сравнения на скрытом тесте (Private LB) лучший одиночный результат показала EfficientNet-B0, она и выбрана как финальная модель.<br /><br />
        EfficientNet-B0 (свёрточная сеть, предобученная на ImageNet) с собственной классификационной головой: Dropout(0.3) + Linear на 6 выходов. Так как задача мультилейбл, на выходе применяется Sigmoid (независимая вероятность для каждого класса), а не Softmax. Функция потерь — BCEWithLogitsLoss с pos_weight для редких классов, чтобы модель не игнорировала их. Параметры финальной модели: 5,3 млн весов.
      </>
    )
  },
  {
    title: 'Обучение',
    text: 'Платформа: Kaggle Notebooks, GPU Tesla T4. Фреймворк: PyTorch + torchvision. Оптимизатор AdamW (lr=3e-4, weight_decay=1e-4), расписание LR — косинусное затухание, 12 эпох, IMG_SIZE=256, batch=64, AMP (mixed precision). Аугментации против переобучения: RandomResizedCrop, горизонтальный/вертикальный флип, поворот, ColorJitter. Разбиение train/val — стратифицированное по комбинации меток, чтобы редкие классы попадали в валидацию пропорционально.'
  },
  {
    title: 'Результаты (EfficientNet-B0)',
    text: (
      <>
        Метрика соревнования — Mean F1-Score.<br />
        Private Leaderboard: <b>0.82349</b> — основная, самая важная оценка (закрытая часть скрытого теста; определяет итоговый результат)<br />
        Public Leaderboard: <b>0.81720</b><br /><br />
        Для сравнения, ансамбль трёх моделей (EfficientNet-B0 + ConvNeXt-Tiny + Swin-Tiny) с усреднением вероятностей дал Private <b>0.82781</b>, однако при кратно большей стоимости инференса (~×20–70 по вычислениям), поэтому в качестве рабочей выбрана одиночная EfficientNet-B0.
      </>
    )
  },
  {
    title: 'Инференс',
    text: 'Изображение масштабируется до 256×256, нормализуется по статистикам ImageNet. Применяется TTA (test-time augmentation): каждый лист прогоняется через модель в трёх видах — оригинал, горизонтальный и вертикальный флип, — а вероятности усредняются. Усреднение убирает чувствительность модели к повороту и даёт небольшой прирост. Итоговые метки формируются по подобранным на валидации порогам для каждого класса (вместо фиксированного 0.5), что особенно улучшает распознавание редких заболеваний. Модель возвращает для каждого изображения набор вероятностей по 6 классам и итоговый список выявленных заболеваний листа (при отсутствии уверенных предсказаний выбирается класс с максимальной вероятностью).'
  },
  {
    title: 'Деплой',
    text: 'Модель упакована в Docker-контейнер с FastAPI-сервисом инференса на PyTorch. Рядом развёрнут Node.js бэкенд на Express, который раздаёт галерею тестовых фото и проксирует запросы к инференсу. Фронтенд на React загружает фото (или снимок с камеры) и показывает найденные болезни с их вероятностями.'
  }
];

export default function App() {
  const [gallery, setGallery] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [resultImage, setResultImage] = useState(null);   // results block only — never shown in the upload card
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingFilename, setLoadingFilename] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [isMobile, setIsMobile] = useState(
    typeof window !== 'undefined' && window.matchMedia('(max-width: 860px)').matches
  );
  const [resultModalOpen, setResultModalOpen] = useState(false);

  // Track viewport width
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 860px)');
    const onChange = e => setIsMobile(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  // Auto-open result modal on mobile when a new result arrives
  useEffect(() => {
    if (result && isMobile) setResultModalOpen(true);
  }, [result, isMobile]);

  // Camera
  const [cameraActive, setCameraActive] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/api/gallery`)
      .then(r => r.json())
      .then(data => setGallery(data.images || []))
      .catch(() => setGallery([]));
  }, []);

  // Stop camera stream on unmount
  useEffect(() => {
    return () => stopCamera();
  }, []);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  };

  const startCamera = async () => {
    setError(null);
    setSelectedFile(null);
    setFilePreview(null);
    setResultImage(null);
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      setCameraActive(true);
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      }, 50);
    } catch {
      setError('Нет доступа к камере');
    }
  };

  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(blob => {
      const file = new File([blob], 'camera.jpg', { type: 'image/jpeg' });
      const url = URL.createObjectURL(file);
      stopCamera();
      setSelectedFile(file);
      setResultImage(url);
      setResult(null);
      setError(null);
      detectWithFile(file);
    }, 'image/jpeg', 0.92);
  };

  const detectWithFile = async (file) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append('image', file);
      const response = await fetch(`${API}/api/detect`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || `HTTP ${response.status}`);
      setResult(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const applyFile = useCallback((file) => {
    const url = URL.createObjectURL(file);
    setSelectedFile(file);
    setResult(null);
    setError(null);
    setResultImage(url);
    detectWithFile(file);
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) { stopCamera(); applyFile(file); }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) { stopCamera(); applyFile(file); }
  };

  const detectGallery = async (filename) => {
    setLoadingFilename(filename);
    setError(null);
    setResult(null);
    setSelectedFile(null);
    setResultImage(`${API}/api/gallery/${encodeURIComponent(filename)}`);
    stopCamera();
    try {
      const response = await fetch(`${API}/api/detect-gallery/${encodeURIComponent(filename)}`, { method: 'POST' });
      if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || `HTTP ${response.status}`);
      setResult(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingFilename(null);
    }
  };

  return (
    <div className="app">
      <button className="info-btn" onClick={() => setShowModal(true)}>
        Как модель создавалась?
      </button>

      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Как создавалась модель</h2>
              <button className="modal-close" onClick={() => setShowModal(false)}>✕</button>
            </div>
            <div className="modal-body">
              {ML_DESCRIPTION.map((section, i) => (
                <div key={i} className="modal-section">
                  <div className="modal-section-number">{i + 1}</div>
                  <div className="modal-section-content">
                    <h3>{section.title}</h3>
                    <p>{section.text}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <header className="header">
        <span className="header-emoji">🍏</span>
        <div>
          <h1>Определение болезней листьев яблони</h1>
          <p>Мультилейбл-классификация болезней на фото листа (EfficientNet-B0)</p>
        </div>
      </header>

      <main className="main">
        <div className="top-row">

          {/* Upload card */}
          <section className="card upload-card">
            <h2>Загрузить фото листа</h2>

            {/* Camera view */}
            {cameraActive && (
              <div className="camera-wrap">
                <video ref={videoRef} className="camera-video" autoPlay playsInline muted />
                <div className="camera-controls">
                  <button className="cam-btn capture-btn" onClick={capturePhoto}>📸 Снять</button>
                  <button className="cam-btn cancel-btn" onClick={stopCamera}>✕ Отмена</button>
                </div>
              </div>
            )}

            {/* Drop zone — the picked photo is shown only in the results block */}
            {!cameraActive && (
              <label
                className={`upload-area ${dragOver ? 'drag-over' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
              >
                <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
                <span className="upload-icon">🍃</span>
                <p>Перетащите или нажмите для выбора</p>
                <p className="upload-hint">JPG, PNG, WebP · до 20 МБ</p>
              </label>
            )}

            {/* Action buttons */}
            <div className="upload-actions">
              {!cameraActive && (
                <button className="cam-open-btn" onClick={startCamera} disabled={loading || !!loadingFilename}>
                  📷 Сделать фото камерой
                </button>
              )}
              {loading && !loadingFilename && (
                <div className="detecting-indicator"><span className="spinner" /> Анализ...</div>
              )}
            </div>

            {error && <div className="error-msg">⚠ {error}</div>}
          </section>

          {/* Gallery card */}
          <section className="card gallery-card">
            <h2>Галерея тестовых фото</h2>
            {gallery.length === 0 ? (
              <p className="muted">Загрузка...</p>
            ) : (
              <div className="gallery">
                {gallery.map(filename => (
                  <button
                    key={filename}
                    className={`gallery-item ${loadingFilename === filename ? 'loading' : ''}`}
                    onClick={() => detectGallery(filename)}
                    disabled={!!loadingFilename || loading || cameraActive}
                    title={filename}
                  >
                    <img
                      src={`${API}/api/gallery/${encodeURIComponent(filename)}`}
                      alt={filename}
                      loading="lazy"
                    />
                    {loadingFilename === filename && <span className="gallery-spinner" />}
                  </button>
                ))}
              </div>
            )}
          </section>
        </div>

        {/* Results — inline (desktop only) */}
        {result && !isMobile && (
          <section className="card results-card">
            <h2>Результат диагностики</h2>
            <ResultBody result={result} preview={resultImage} />
          </section>
        )}
      </main>

      {/* Results — modal (mobile) */}
      {result && isMobile && resultModalOpen && (
        <div className="modal-overlay" onClick={() => setResultModalOpen(false)}>
          <div className="modal results-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Результат диагностики</h2>
              <button className="modal-close" onClick={() => setResultModalOpen(false)}>✕</button>
            </div>
            <div className="modal-body">
              <ResultBody result={result} preview={resultImage} />
            </div>
          </div>
        </div>
      )}

      {/* Hidden canvas for photo capture */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
}

function ResultBody({ result, preview }) {
  const healthy = result.is_healthy;
  const diseases = result.diseases || [];
  // "Комплекс болезней" (complex) — keep its Grad-CAM tab last.
  const gradcam = (result.gradcam || []).slice().sort(
    (a, b) => (a.label === 'complex') - (b.label === 'complex')
  );

  const [camIdx, setCamIdx] = useState(0);
  useEffect(() => { setCamIdx(0); }, [result]);
  const cam = gradcam[Math.min(camIdx, Math.max(gradcam.length - 1, 0))];

  return (
    <div className="results-wrap">
      {/* Top: photo (left) + detected diseases with descriptions (right) */}
      <div className="diag-row">
        {preview && (
          <div className="result-img-wrap">
            <img src={preview} alt="Лист" className="result-img" />
          </div>
        )}

        <div className="diag-info">
          <div className={`verdict-title ${healthy ? 'verdict-healthy' : 'verdict-sick'}`}>
            {healthy ? 'Лист здоров' : 'Обнаружены болезни'}
          </div>

          {healthy ? (
            <p className="healthy-note">Признаков болезней не обнаружено.</p>
          ) : (
            <div className="disease-list">
              {diseases.map(d => (
                <div key={d.label} className="disease-item">
                  <div className="disease-head">
                    <span className="disease-name">{d.name_ru}</span>
                  </div>
                  {d.description && <div className="disease-desc">{d.description}</div>}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Bottom: extra info — Grad-CAM (left) + all-class probabilities (right) */}
      <div className="extra">
        <h3 className="extra-title">Дополнительная информация</h3>
        <div className="extra-row">
          {cam && (
            <div className="gradcam">
              <div className="gradcam-title">🔍 Куда смотрела модель</div>
              {gradcam.length > 1 && (
                <div className="gradcam-tabs">
                  {gradcam.map((g, i) => (
                    <button
                      key={g.label}
                      className={`gradcam-tab ${i === camIdx ? 'active' : ''}`}
                      onClick={() => setCamIdx(i)}
                    >
                      {g.name_ru}
                    </button>
                  ))}
                </div>
              )}
              <div className="gradcam-img-wrap">
                <img
                  src={`data:image/jpeg;base64,${cam.image}`}
                  alt={`Grad-CAM: ${cam.name_ru}`}
                  className="gradcam-img"
                />
              </div>
              <div className="gradcam-hint">
                Красным выделены участки, сильнее всего повлиявшие на решение по классу
                «{cam.name_ru}», синим — области с малым вкладом.
              </div>
            </div>
          )}

          <div className="breakdown">
            <div className="breakdown-title">Вероятности по всем классам</div>
            {result.predictions.map(p => (
              <div key={p.label} className={`bar-row ${p.detected ? 'bar-on' : ''}`}>
                <div className="bar-label">
                  <span>{p.name_ru}</span>
                  <span className="bar-pct">{(p.probability * 100).toFixed(0)}%</span>
                </div>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${Math.min(p.probability * 100, 100)}%` }} />
                  <div className="bar-threshold" style={{ left: `${Math.min(p.threshold * 100, 100)}%` }} title={`порог ${(p.threshold * 100).toFixed(0)}%`} />
                </div>
              </div>
            ))}
            <div className="breakdown-hint">
              Вертикальная риска — порог класса. Класс попадает в диагноз, если вероятность превышает свой порог.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
