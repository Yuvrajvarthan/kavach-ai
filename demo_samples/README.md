# Demo samples

All messages here are **fictional**, written for testing Kavach. Phone numbers (+91 90000 …),
bank names ("XYZ Bank") and links (`*.example`) are deliberately fake.
They mirror the scam patterns Indian banks and cyber cells warn about.

- `messages.json`: 14 test messages (8 scam across the 4 scam classes, 2 Hinglish, 4 legit) with expected labels
- `screenshot_*.png`: phone-style screenshots for the Screenshot (OCR) tab
- `make_audio.sh`: run on macOS to create voice clips with the built-in `say` voices

Viva demo order: gov_01 (text) → screenshot_parcel_01 (OCR) → audio_query_hinglish (assistant) → legit_01 (shows no false alarm).
