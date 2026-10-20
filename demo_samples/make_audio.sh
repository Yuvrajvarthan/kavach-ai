#!/bin/bash
# Run on the Mac (uses the built-in `say` voices). Creates demo voice clips as .wav for the Voice tab.
# If a voice is missing: System Settings > Accessibility > Spoken Content > System voice > Manage Voices.
cd "$(dirname "$0")"
say -v Rishi -o audio_gov_01.aiff "This is Mumbai Cyber Crime Cell. A parcel booked on your Aadhaar contains illegal drugs. An F I R has been registered. Join the video call within thirty minutes or an arrest warrant will be issued."
say -v Rishi -o audio_query_en.aiff "I got a message saying my bank K Y C has expired and my account will be blocked today. Is this a scam?"
say -v Lekha -o audio_query_hinglish.aiff "Mujhe message aaya hai ki mera bijli connection aaj raat kaat diya jayega. Kya yeh scam hai?"
for f in *.aiff; do afconvert -f WAVE -d LEI16@16000 "$f" "${f%.aiff}.wav" && rm "$f"; done
ls -1 *.wav
