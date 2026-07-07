const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3002;
const INFERENCE_URL = process.env.INFERENCE_URL || 'http://apple-inference:8001';
const GALLERY_DIR = process.env.GALLERY_DIR || '/app/test_photos';

const IMAGE_EXTS = /\.(jpg|jpeg|png|webp)$/i;

app.use(cors());
app.use(express.json());

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    cb(null, IMAGE_EXTS.test(file.originalname));
  },
});

app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.get('/api/gallery', (req, res) => {
  try {
    const files = fs.readdirSync(GALLERY_DIR).filter(f => IMAGE_EXTS.test(f));
    res.json({ images: files });
  } catch (err) {
    res.status(500).json({ error: 'Cannot read gallery directory' });
  }
});

app.get('/api/gallery/:filename', (req, res) => {
  const filename = path.basename(req.params.filename);
  const filepath = path.join(GALLERY_DIR, filename);
  if (!fs.existsSync(filepath)) {
    return res.status(404).json({ error: 'Not found' });
  }
  res.sendFile(filepath);
});

app.post('/api/detect', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image provided' });
  }
  try {
    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });
    const response = await axios.post(`${INFERENCE_URL}/detect`, form, {
      headers: form.getHeaders(),
      timeout: 120000,
    });
    res.json(response.data);
  } catch (err) {
    const msg = err.response?.data?.detail || err.response?.data?.error || err.message;
    res.status(500).json({ error: msg });
  }
});

app.post('/api/detect-gallery/:filename', async (req, res) => {
  const filename = path.basename(req.params.filename);
  const filepath = path.join(GALLERY_DIR, filename);
  if (!fs.existsSync(filepath)) {
    return res.status(404).json({ error: 'Not found' });
  }
  try {
    const form = new FormData();
    form.append('file', fs.createReadStream(filepath), { filename });
    const response = await axios.post(`${INFERENCE_URL}/detect`, form, {
      headers: form.getHeaders(),
      timeout: 120000,
    });
    res.json(response.data);
  } catch (err) {
    const msg = err.response?.data?.detail || err.response?.data?.error || err.message;
    res.status(500).json({ error: msg });
  }
});

app.listen(PORT, () => console.log(`Apple-tree-diseases backend running on port ${PORT}`));
