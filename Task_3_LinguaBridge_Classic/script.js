(function () {
  'use strict';

  /* ---------- Settings ---------- */
  var API_URL = 'https://api.mymemory.translated.net/get';
  // Optional: put your email here to raise the free daily limit (about 50,000 characters).
  var CONTACT_EMAIL = '';
  var MAX_CHARS = 5000;
  var CHUNK_SIZE = 450;      // MyMemory accepts at most 500 bytes per request
  var TIMEOUT_MS = 15000;

  var LANGUAGES = [
    ['ar', 'Arabic'], ['bn', 'Bengali'], ['zh-CN', 'Chinese (Simplified)'], ['zh-TW', 'Chinese (Traditional)'],
    ['nl', 'Dutch'], ['en', 'English'], ['fr', 'French'], ['de', 'German'], ['gu', 'Gujarati'],
    ['hi', 'Hindi'], ['id', 'Indonesian'], ['it', 'Italian'], ['ja', 'Japanese'], ['kn', 'Kannada'],
    ['ko', 'Korean'], ['ml', 'Malayalam'], ['mr', 'Marathi'], ['ne', 'Nepali'], ['fa', 'Persian'],
    ['pl', 'Polish'], ['pt', 'Portuguese'], ['pa', 'Punjabi'], ['ru', 'Russian'], ['es', 'Spanish'],
    ['sw', 'Swahili'], ['ta', 'Tamil'], ['te', 'Telugu'], ['th', 'Thai'], ['tr', 'Turkish'],
    ['uk', 'Ukrainian'], ['ur', 'Urdu'], ['vi', 'Vietnamese']
  ];
  var AUTO = 'Autodetect';

  /* ---------- Pure helpers ---------- */
  function langName(code) {
    for (var i = 0; i < LANGUAGES.length; i++) {
      if (LANGUAGES[i][0].toLowerCase() === String(code).toLowerCase()) return LANGUAGES[i][1];
    }
    return String(code);
  }

  function hasLang(code) {
    return langName(code) !== String(code);
  }

  function splitLongSentence(sentence, size) {
    var parts = [];
    var current = '';
    var words = sentence.split(/(\s+)/);
    for (var i = 0; i < words.length; i++) {
      var w = words[i];
      while (w.length > size) {            // a single huge "word" (e.g. no spaces)
        if (current) { parts.push(current); current = ''; }
        parts.push(w.slice(0, size));
        w = w.slice(size);
      }
      if ((current + w).length > size) { parts.push(current); current = w; }
      else { current += w; }
    }
    if (current) parts.push(current);
    return parts;
  }

  // Returns an array of lines; each line is an array of chunks (<= size chars).
  function splitText(text, size) {
    var lines = text.replace(/\r\n?/g, '\n').split('\n');
    return lines.map(function (line) {
      if (!line.trim()) return [];
      var sentences = line.match(/[^.!?\u0964\u3002\uFF01\uFF1F]+[.!?\u0964\u3002\uFF01\uFF1F]*\s*/g) || [line];
      var chunks = [];
      var current = '';
      sentences.forEach(function (s) {
        if (s.length > size) {
          if (current) { chunks.push(current); current = ''; }
          splitLongSentence(s, size).forEach(function (p) { chunks.push(p); });
        } else if ((current + s).length > size) {
          chunks.push(current);
          current = s;
        } else {
          current += s;
        }
      });
      if (current) chunks.push(current);
      return chunks;
    });
  }

  function decodeEntities(str) {
    var doc = new DOMParser().parseFromString(str, 'text/html');
    return doc.documentElement.textContent || '';
  }

  /* ---------- API ---------- */
  async function translateChunk(chunk, src, tgt) {
    var params = new URLSearchParams({ q: chunk, langpair: src + '|' + tgt });
    if (CONTACT_EMAIL) params.set('de', CONTACT_EMAIL);

    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, TIMEOUT_MS);
    try {
      var res = await fetch(API_URL + '?' + params.toString(), { signal: controller.signal });
      if (!res.ok) throw new Error('The translation service returned an error (' + res.status + '). Try again shortly.');
      var data = await res.json();
      var status = Number(data.responseStatus);
      var text = data.responseData && data.responseData.translatedText;

      if (status !== 200 || typeof text !== 'string') {
        var details = String(data.responseDetails || '');
        if (/distinct/i.test(details)) throw new Error('The source and target languages are the same. Choose a different target language.');
        if (status === 429) throw new Error('Free daily limit reached. Try again later or add your email in script.js.');
        throw new Error(details || 'Translation failed. Please try again.');
      }
      if (/MYMEMORY WARNING/i.test(text)) {
        throw new Error('Free daily limit reached. Try again later or add your email in script.js.');
      }
      return {
        text: decodeEntities(text),
        detected: (data.responseData && data.responseData.detectedLanguage) || null
      };
    } catch (err) {
      if (err && err.name === 'AbortError') throw new Error('The request timed out. Check your connection and try again.');
      if (err instanceof TypeError) throw new Error('Cannot reach the translation service. Check your internet connection.');
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  async function translateText(text, src, tgt) {
    var lines = splitText(text, CHUNK_SIZE);
    var detected = null;
    var out = [];
    for (var i = 0; i < lines.length; i++) {
      var pieces = [];
      for (var j = 0; j < lines[i].length; j++) {
        var r = await translateChunk(lines[i][j], src, tgt);
        if (!detected && r.detected) detected = r.detected;
        pieces.push(r.text);
      }
      out.push(pieces.join(''));
    }
    return { text: out.join('\n'), detected: detected };
  }

  /* ---------- UI ---------- */
  if (typeof document === 'undefined') {
    // Allows the helpers to be tested outside a browser.
    if (typeof module !== 'undefined') module.exports = { splitText: splitText };
    return;
  }

  var el = {
    source: document.getElementById('source'),
    target: document.getElementById('target'),
    input: document.getElementById('input'),
    output: document.getElementById('output'),
    counter: document.getElementById('counter'),
    detected: document.getElementById('detected'),
    status: document.getElementById('status'),
    translate: document.getElementById('translate'),
    clear: document.getElementById('clear'),
    swap: document.getElementById('swap'),
    copy: document.getElementById('copy'),
    speak: document.getElementById('speak'),
    stop: document.getElementById('stop'),
    paste: document.getElementById('paste')
  };

  var lastResult = '';
  var lastTarget = 'te';
  var lastDetected = null;
  var busy = false;
  var speaking = false;
  var speakToken = 0;
  var audioEl = null;
  var canSpeak = 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;

  function fillSelect(select, withAuto, selected) {
    var html = '';
    if (withAuto) html += '<option value="' + AUTO + '">Detect language</option>';
    LANGUAGES.forEach(function (l) { html += '<option value="' + l[0] + '">' + l[1] + '</option>'; });
    select.innerHTML = html;
    select.value = selected;
  }

  function setStatus(msg, isError) {
    el.status.textContent = msg || '';
    el.status.className = 'status' + (isError ? ' is-error' : '');
  }

  function showPlaceholder() {
    el.output.className = 'output';
    el.output.innerHTML = '<span class="placeholder">The translation appears here.</span>';
  }

  function clearResult() {
    lastResult = '';
    lastDetected = null;
    el.detected.textContent = '';
    showPlaceholder();
    refreshButtons();
    stopSpeaking();
  }

  function refreshButtons() {
    var hasText = el.input.value.trim().length > 0;
    el.translate.disabled = busy || !hasText;
    el.clear.disabled = el.input.value.length === 0;
    el.copy.disabled = !lastResult;
    el.speak.disabled = !lastResult || speaking;
    el.stop.disabled = !speaking;
    el.swap.disabled = busy;
    el.counter.textContent = el.input.value.length + ' / ' + MAX_CHARS;
  }

  async function doTranslate() {
    if (busy) return;
    var text = el.input.value.trim();
    if (!text) return;
    var src = el.source.value;
    var tgt = el.target.value;

    stopSpeaking();
    busy = true;
    refreshButtons();
    el.translate.textContent = 'Translating…';
    el.output.className = 'output is-loading';
    setStatus('Translating…', false);

    try {
      if (src === tgt) {
        lastResult = text;
        lastDetected = null;
      } else {
        var r = await translateText(text, src, tgt);
        lastResult = r.text;
        lastDetected = src === AUTO ? r.detected : null;
      }
      lastTarget = tgt;
      el.output.className = 'output';
      el.output.textContent = lastResult;
      el.detected.textContent = lastDetected && hasLang(lastDetected) ? 'Detected: ' + langName(lastDetected) : '';
      setStatus('Done.', false);
    } catch (err) {
      lastResult = '';
      lastDetected = null;
      el.detected.textContent = '';
      el.output.className = 'output is-error';
      el.output.textContent = err.message || 'Something went wrong. Please try again.';
      setStatus('Translation failed.', true);
    } finally {
      busy = false;
      el.translate.textContent = 'Translate';
      refreshButtons();
    }
  }

  function swapLanguages() {
    var src = el.source.value;
    var tgt = el.target.value;
    var newTarget = src;
    if (src === AUTO) {
      newTarget = lastDetected && hasLang(lastDetected) ? lastDetected : null;
      if (!newTarget) { setStatus('Choose a source language first, then swap.', true); return; }
    }
    var oldOutput = lastResult;
    el.source.value = tgt;
    el.target.value = newTarget;
    if (oldOutput) el.input.value = oldOutput;
    clearResult();
    refreshButtons();
    setStatus('', false);
    if (el.input.value.trim()) doTranslate();
  }

  async function copyResult() {
    if (!lastResult) return;
    var ok = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(lastResult);
        ok = true;
      }
    } catch (e) { ok = false; }
    if (!ok) {
      var ta = document.createElement('textarea');
      ta.value = lastResult;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { ok = document.execCommand('copy'); } catch (e2) { ok = false; }
      document.body.removeChild(ta);
    }
    el.copy.textContent = ok ? 'Copied' : 'Copy failed';
    setStatus(ok ? 'Translation copied to clipboard.' : 'Could not copy. Select the text and press Ctrl + C.', !ok);
    setTimeout(function () { el.copy.textContent = 'Copy'; }, 1500);
  }

  function setSpeaking(on) {
    speaking = on;
    refreshButtons();
  }

  function stopSpeaking() {
    speakToken++;                       // ignore events from the speech we are cancelling
    if (audioEl) {
      audioEl.onended = null;
      audioEl.onerror = null;
      audioEl.pause();
      audioEl = null;
    }
    if (canSpeak) window.speechSynthesis.cancel();
    setSpeaking(false);
  }

  // Finds a voice installed in this browser/PC for the language (or null).
  function findVoice(lang) {
    if (!canSpeak) return null;
    var voices = [];
    try { voices = window.speechSynthesis.getVoices() || []; } catch (e) { voices = []; }
    var full = lang.toLowerCase();
    var base = full.split('-')[0];
    for (var i = 0; i < voices.length; i++) {
      var vl = String(voices[i].lang || '').toLowerCase().replace('_', '-');
      if (vl === full || vl.split('-')[0] === base) return voices[i];
    }
    return null;
  }

  function speakWithBrowser(parts, voice, token) {
    parts.forEach(function (piece, i) {
      var u = new SpeechSynthesisUtterance(piece);
      u.voice = voice;
      u.lang = voice.lang;
      u.onend = function () {
        if (token === speakToken && i === parts.length - 1) setSpeaking(false);
      };
      u.onerror = function (e) {
        if (token !== speakToken) return;
        var reason = e && e.error;
        if (reason === 'canceled' || reason === 'interrupted') return;
        stopSpeaking();
        setStatus('Could not play audio in this browser.', true);
      };
      window.speechSynthesis.speak(u);
    });
  }

  // Online audio (works for languages with no voice installed on the PC).
  function speakOnline(parts, lang, token) {
    var index = 0;
    function fail() {
      if (token !== speakToken) return;
      stopSpeaking();
      setStatus('Could not load audio. Check your internet connection and try again.', true);
    }
    function next() {
      if (token !== speakToken) return;
      if (index >= parts.length) { audioEl = null; setSpeaking(false); return; }
      var url = 'https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=' +
        encodeURIComponent(lang) + '&q=' + encodeURIComponent(parts[index]);
      index++;
      var a = new Audio();
      audioEl = a;
      a.onended = next;
      a.onerror = fail;
      a.src = url;
      var p = a.play();
      if (p && typeof p.catch === 'function') p.catch(fail);
    }
    next();
  }

  function speakResult() {
    if (!lastResult || speaking) return;
    stopSpeaking();

    var voice = findVoice(lastTarget);
    var parts = [];
    splitText(lastResult, voice ? 200 : 180).forEach(function (line) {
      line.forEach(function (c) { if (c.trim()) parts.push(c.trim()); });
    });
    if (!parts.length) return;

    var token = ++speakToken;
    setSpeaking(true);
    setStatus('', false);
    if (voice) speakWithBrowser(parts, voice, token);
    else speakOnline(parts, lastTarget, token);
  }

  async function pasteText() {
    var text = '';
    try {
      if (!navigator.clipboard || !navigator.clipboard.readText) throw new Error('unsupported');
      text = await navigator.clipboard.readText();
    } catch (e) {
      el.input.focus();
      setStatus('Browser blocked paste. Click in the text box and press Ctrl + V.', true);
      return;
    }
    if (!text) { setStatus('Clipboard is empty.', true); return; }

    var start = el.input.selectionStart || 0;
    var end = el.input.selectionEnd || 0;
    var value = el.input.value;
    var next = value.slice(0, start) + text + value.slice(end);
    var trimmed = next.length > MAX_CHARS;
    if (trimmed) next = next.slice(0, MAX_CHARS);
    el.input.value = next;
    var caret = Math.min(start + text.length, next.length);
    el.input.focus();
    el.input.setSelectionRange(caret, caret);
    refreshButtons();
    setStatus(trimmed ? 'Pasted. Text was cut to ' + MAX_CHARS + ' characters.' : 'Pasted.', trimmed);
  }

  /* ---------- Wire up ---------- */
  fillSelect(el.source, true, 'en');
  fillSelect(el.target, false, 'te');

  el.input.addEventListener('input', function () { refreshButtons(); });
  el.input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doTranslate(); }
  });
  el.translate.addEventListener('click', doTranslate);
  el.swap.addEventListener('click', swapLanguages);
  el.copy.addEventListener('click', copyResult);
  el.speak.addEventListener('click', speakResult);
  el.stop.addEventListener('click', stopSpeaking);
  el.paste.addEventListener('click', pasteText);
  el.clear.addEventListener('click', function () {
    el.input.value = '';
    clearResult();
    setStatus('', false);
    el.input.focus();
  });
  el.source.addEventListener('change', clearResult);
  el.target.addEventListener('change', clearResult);

  refreshButtons();
})();
