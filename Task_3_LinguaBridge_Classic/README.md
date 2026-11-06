# LinguaBridge – Language Translation Tool

A web app that translates text between 30+ languages.

## How to run
(Windows shortcut: double-click `START.bat`, then press Ctrl + Shift + R in the browser.)

1. Unzip the folder.
2. Double-click `index.html` (use Chrome, Edge or Firefox).
3. An internet connection is required (the app calls the translation API).

No installation, no API key, no server needed.

## Features (matches the task list)
- User interface: text box, source and target language dropdowns ("Detect language" is available for the source)
- Uses the free **MyMemory Translation API** to translate the text
- Long text is split into safe chunks, so large inputs also work
- Translated text is displayed clearly on screen
- Optional extras: **Copy** and **Paste** buttons, **Listen** and **Stop** (text-to-speech), **Swap languages**, character counter, Ctrl + Enter shortcut
- Friendly error messages (offline, timeout, daily limit reached)

## Files
- `index.html` – page structure
- `style.css` – styling (responsive, works on mobile)
- `script.js` – translation logic, API call, copy and speech features

## Notes
- The free API allows about 5,000 characters per day without any setup. To raise it to about 50,000, put your email in `CONTACT_EMAIL` at the top of `script.js`.
- Listen uses a voice installed on your PC when one exists for the language; otherwise it plays online audio (internet required). Press Stop to end it.
- To use Google Translate or Microsoft Translator instead, replace `translateChunk()` in `script.js` with a call to that API (they require an API key).
