#!/usr/bin/env python3
"""
ViralTTS by EFXTv — 322 Edge neural voices organized by language.

USAGE:
    python app.py list
    python app.py list English
    python app.py list Hindi
    python app.py list Spanish
    python app.py <voice-selector> script.txt [output.wav|output.mp3] [--style STYLE]

EXAMPLES:
    python app.py brian-multilingual script.txt voiceover.mp3 --style narrative
    python app.py aria script.txt narration.wav --style emotional
    python app.py madhur hindi.txt hindi.mp3 --style deep
    python app.py swara hindi.txt hindi.mp3 --style emotional
    python app.py jorge spanish.txt spanish.mp3 --style warm

STYLES:
    natural, narrative, deep, emotional, warm, cinematic

Internet is required. Microsoft Edge neural TTS is used without an API key.
For Termux, install FFmpeg first: pkg install python ffmpeg
"""

import argparse
import asyncio
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

print(r"""
╔══════════════════════════════════════════╗
║                                          ║
║          ViralTTS by EFXTv               ║
║                                          ║
╚══════════════════════════════════════════╝
""")

# Complete 322-voice catalog from the supplied Edge TTS voice list.
# Format: Edge voice ID | gender | content categories | personality
VOICE_CATALOG = r"""
af-ZA-AdriNeural|Female|General|Friendly, Positive
af-ZA-WillemNeural|Male|General|Friendly, Positive
am-ET-AmehaNeural|Male|General|Friendly, Positive
am-ET-MekdesNeural|Female|General|Friendly, Positive
ar-AE-FatimaNeural|Female|General|Friendly, Positive
ar-AE-HamdanNeural|Male|General|Friendly, Positive
ar-BH-AliNeural|Male|General|Friendly, Positive
ar-BH-LailaNeural|Female|General|Friendly, Positive
ar-DZ-AminaNeural|Female|General|Friendly, Positive
ar-DZ-IsmaelNeural|Male|General|Friendly, Positive
ar-EG-SalmaNeural|Female|General|Friendly, Positive
ar-EG-ShakirNeural|Male|General|Friendly, Positive
ar-IQ-BasselNeural|Male|General|Friendly, Positive
ar-IQ-RanaNeural|Female|General|Friendly, Positive
ar-JO-SanaNeural|Female|General|Friendly, Positive
ar-JO-TaimNeural|Male|General|Friendly, Positive
ar-KW-FahedNeural|Male|General|Friendly, Positive
ar-KW-NouraNeural|Female|General|Friendly, Positive
ar-LB-LaylaNeural|Female|General|Friendly, Positive
ar-LB-RamiNeural|Male|General|Friendly, Positive
ar-LY-ImanNeural|Female|General|Friendly, Positive
ar-LY-OmarNeural|Male|General|Friendly, Positive
ar-MA-JamalNeural|Male|General|Friendly, Positive
ar-MA-MounaNeural|Female|General|Friendly, Positive
ar-OM-AbdullahNeural|Male|General|Friendly, Positive
ar-OM-AyshaNeural|Female|General|Friendly, Positive
ar-QA-AmalNeural|Female|General|Friendly, Positive
ar-QA-MoazNeural|Male|General|Friendly, Positive
ar-SA-HamedNeural|Male|General|Friendly, Positive
ar-SA-ZariyahNeural|Female|General|Friendly, Positive
ar-SY-AmanyNeural|Female|General|Friendly, Positive
ar-SY-LaithNeural|Male|General|Friendly, Positive
ar-TN-HediNeural|Male|General|Friendly, Positive
ar-TN-ReemNeural|Female|General|Friendly, Positive
ar-YE-MaryamNeural|Female|General|Friendly, Positive
ar-YE-SalehNeural|Male|General|Friendly, Positive
az-AZ-BabekNeural|Male|General|Friendly, Positive
az-AZ-BanuNeural|Female|General|Friendly, Positive
bg-BG-BorislavNeural|Male|General|Friendly, Positive
bg-BG-KalinaNeural|Female|General|Friendly, Positive
bn-BD-NabanitaNeural|Female|General|Friendly, Positive
bn-BD-PradeepNeural|Male|General|Friendly, Positive
bn-IN-BashkarNeural|Male|General|Friendly, Positive
bn-IN-TanishaaNeural|Female|General|Friendly, Positive
bs-BA-GoranNeural|Male|General|Friendly, Positive
bs-BA-VesnaNeural|Female|General|Friendly, Positive
ca-ES-EnricNeural|Male|General|Friendly, Positive
ca-ES-JoanaNeural|Female|General|Friendly, Positive
cs-CZ-AntoninNeural|Male|General|Friendly, Positive
cs-CZ-VlastaNeural|Female|General|Friendly, Positive
cy-GB-AledNeural|Male|General|Friendly, Positive
cy-GB-NiaNeural|Female|General|Friendly, Positive
da-DK-ChristelNeural|Female|General|Friendly, Positive
da-DK-JeppeNeural|Male|General|Friendly, Positive
de-AT-IngridNeural|Female|General|Friendly, Positive
de-AT-JonasNeural|Male|General|Friendly, Positive
de-CH-JanNeural|Male|General|Friendly, Positive
de-CH-LeniNeural|Female|General|Friendly, Positive
de-DE-AmalaNeural|Female|General|Friendly, Positive
de-DE-ConradNeural|Male|General|Friendly, Positive
de-DE-FlorianMultilingualNeural|Male|General|Friendly, Positive
de-DE-KatjaNeural|Female|General|Friendly, Positive
de-DE-KillianNeural|Male|General|Friendly, Positive
de-DE-SeraphinaMultilingualNeural|Female|General|Friendly, Positive
el-GR-AthinaNeural|Female|General|Friendly, Positive
el-GR-NestorasNeural|Male|General|Friendly, Positive
en-AU-NatashaNeural|Female|General|Friendly, Positive
en-AU-WilliamMultilingualNeural|Male|General|Friendly, Positive
en-CA-ClaraNeural|Female|General|Friendly, Positive
en-CA-LiamNeural|Male|General|Friendly, Positive
en-GB-LibbyNeural|Female|General|Friendly, Positive
en-GB-MaisieNeural|Female|General|Friendly, Positive
en-GB-RyanNeural|Male|General|Friendly, Positive
en-GB-SoniaNeural|Female|General|Friendly, Positive
en-GB-ThomasNeural|Male|General|Friendly, Positive
en-HK-SamNeural|Male|General|Friendly, Positive
en-HK-YanNeural|Female|General|Friendly, Positive
en-IE-ConnorNeural|Male|General|Friendly, Positive
en-IE-EmilyNeural|Female|General|Friendly, Positive
en-IN-NeerjaExpressiveNeural|Female|General|Friendly, Positive
en-IN-NeerjaNeural|Female|General|Friendly, Positive
en-IN-PrabhatNeural|Male|General|Friendly, Positive
en-KE-AsiliaNeural|Female|General|Friendly, Positive
en-KE-ChilembaNeural|Male|General|Friendly, Positive
en-NG-AbeoNeural|Male|General|Friendly, Positive
en-NG-EzinneNeural|Female|General|Friendly, Positive
en-NZ-MitchellNeural|Male|General|Friendly, Positive
en-NZ-MollyNeural|Female|General|Friendly, Positive
en-PH-JamesNeural|Male|General|Friendly, Positive
en-PH-RosaNeural|Female|General|Friendly, Positive
en-SG-LunaNeural|Female|General|Friendly, Positive
en-SG-WayneNeural|Male|General|Friendly, Positive
en-TZ-ElimuNeural|Male|General|Friendly, Positive
en-TZ-ImaniNeural|Female|General|Friendly, Positive
en-US-AnaNeural|Female|Cartoon, Conversation|Cute
en-US-AndrewMultilingualNeural|Male|Conversation, Copilot|Warm, Confident, Authentic, Honest
en-US-AndrewNeural|Male|Conversation, Copilot|Warm, Confident, Authentic, Honest
en-US-AriaNeural|Female|News, Novel|Positive, Confident
en-US-AvaMultilingualNeural|Female|Conversation, Copilot|Expressive, Caring, Pleasant, Friendly
en-US-AvaNeural|Female|Conversation, Copilot|Expressive, Caring, Pleasant, Friendly
en-US-BrianMultilingualNeural|Male|Conversation, Copilot|Approachable, Casual, Sincere
en-US-BrianNeural|Male|Conversation, Copilot|Approachable, Casual, Sincere
en-US-ChristopherNeural|Male|News, Novel|Reliable, Authority
en-US-EmmaMultilingualNeural|Female|Conversation, Copilot|Cheerful, Clear, Conversational
en-US-EmmaNeural|Female|Conversation, Copilot|Cheerful, Clear, Conversational
en-US-EricNeural|Male|News, Novel|Rational
en-US-GuyNeural|Male|News, Novel|Passion
en-US-JennyNeural|Female|General|Friendly, Considerate, Comfort
en-US-MichelleNeural|Female|News, Novel|Friendly, Pleasant
en-US-RogerNeural|Male|News, Novel|Lively
en-US-SteffanNeural|Male|News, Novel|Rational
en-ZA-LeahNeural|Female|General|Friendly, Positive
en-ZA-LukeNeural|Male|General|Friendly, Positive
es-AR-ElenaNeural|Female|General|Friendly, Positive
es-AR-TomasNeural|Male|General|Friendly, Positive
es-BO-MarceloNeural|Male|General|Friendly, Positive
es-BO-SofiaNeural|Female|General|Friendly, Positive
es-CL-CatalinaNeural|Female|General|Friendly, Positive
es-CL-LorenzoNeural|Male|General|Friendly, Positive
es-CO-GonzaloNeural|Male|General|Friendly, Positive
es-CO-SalomeNeural|Female|General|Friendly, Positive
es-CR-JuanNeural|Male|General|Friendly, Positive
es-CR-MariaNeural|Female|General|Friendly, Positive
es-CU-BelkysNeural|Female|General|Friendly, Positive
es-CU-ManuelNeural|Male|General|Friendly, Positive
es-DO-EmilioNeural|Male|General|Friendly, Positive
es-DO-RamonaNeural|Female|General|Friendly, Positive
es-EC-AndreaNeural|Female|General|Friendly, Positive
es-EC-LuisNeural|Male|General|Friendly, Positive
es-ES-AlvaroNeural|Male|General|Friendly, Positive
es-ES-ElviraNeural|Female|General|Friendly, Positive
es-ES-XimenaNeural|Female|General|Friendly, Positive
es-GQ-JavierNeural|Male|General|Friendly, Positive
es-GQ-TeresaNeural|Female|General|Friendly, Positive
es-GT-AndresNeural|Male|General|Friendly, Positive
es-GT-MartaNeural|Female|General|Friendly, Positive
es-HN-CarlosNeural|Male|General|Friendly, Positive
es-HN-KarlaNeural|Female|General|Friendly, Positive
es-MX-DaliaNeural|Female|General|Friendly, Positive
es-MX-JorgeNeural|Male|General|Friendly, Positive
es-NI-FedericoNeural|Male|General|Friendly, Positive
es-NI-YolandaNeural|Female|General|Friendly, Positive
es-PA-MargaritaNeural|Female|General|Friendly, Positive
es-PA-RobertoNeural|Male|General|Friendly, Positive
es-PE-AlexNeural|Male|General|Friendly, Positive
es-PE-CamilaNeural|Female|General|Friendly, Positive
es-PR-KarinaNeural|Female|General|Friendly, Positive
es-PR-VictorNeural|Male|General|Friendly, Positive
es-PY-MarioNeural|Male|General|Friendly, Positive
es-PY-TaniaNeural|Female|General|Friendly, Positive
es-SV-LorenaNeural|Female|General|Friendly, Positive
es-SV-RodrigoNeural|Male|General|Friendly, Positive
es-US-AlonsoNeural|Male|General|Friendly, Positive
es-US-PalomaNeural|Female|General|Friendly, Positive
es-UY-MateoNeural|Male|General|Friendly, Positive
es-UY-ValentinaNeural|Female|General|Friendly, Positive
es-VE-PaolaNeural|Female|General|Friendly, Positive
es-VE-SebastianNeural|Male|General|Friendly, Positive
et-EE-AnuNeural|Female|General|Friendly, Positive
et-EE-KertNeural|Male|General|Friendly, Positive
fa-IR-DilaraNeural|Female|General|Friendly, Positive
fa-IR-FaridNeural|Male|General|Friendly, Positive
fi-FI-HarriNeural|Male|General|Friendly, Positive
fi-FI-NooraNeural|Female|General|Friendly, Positive
fil-PH-AngeloNeural|Male|General|Friendly, Positive
fil-PH-BlessicaNeural|Female|General|Friendly, Positive
fr-BE-CharlineNeural|Female|General|Friendly, Positive
fr-BE-GerardNeural|Male|General|Friendly, Positive
fr-CA-AntoineNeural|Male|General|Friendly, Positive
fr-CA-JeanNeural|Male|General|Friendly, Positive
fr-CA-SylvieNeural|Female|General|Friendly, Positive
fr-CA-ThierryNeural|Male|General|Friendly, Positive
fr-CH-ArianeNeural|Female|General|Friendly, Positive
fr-CH-FabriceNeural|Male|General|Friendly, Positive
fr-FR-DeniseNeural|Female|General|Friendly, Positive
fr-FR-EloiseNeural|Female|General|Friendly, Positive
fr-FR-HenriNeural|Male|General|Friendly, Positive
fr-FR-RemyMultilingualNeural|Male|General|Friendly, Positive
fr-FR-VivienneMultilingualNeural|Female|General|Friendly, Positive
ga-IE-ColmNeural|Male|General|Friendly, Positive
ga-IE-OrlaNeural|Female|General|Friendly, Positive
gl-ES-RoiNeural|Male|General|Friendly, Positive
gl-ES-SabelaNeural|Female|General|Friendly, Positive
gu-IN-DhwaniNeural|Female|General|Friendly, Positive
gu-IN-NiranjanNeural|Male|General|Friendly, Positive
he-IL-AvriNeural|Male|General|Friendly, Positive
he-IL-HilaNeural|Female|General|Friendly, Positive
hi-IN-MadhurNeural|Male|General|Friendly, Positive
hi-IN-SwaraNeural|Female|General|Friendly, Positive
hr-HR-GabrijelaNeural|Female|General|Friendly, Positive
hr-HR-SreckoNeural|Male|General|Friendly, Positive
hu-HU-NoemiNeural|Female|General|Friendly, Positive
hu-HU-TamasNeural|Male|General|Friendly, Positive
id-ID-ArdiNeural|Male|General|Friendly, Positive
id-ID-GadisNeural|Female|General|Friendly, Positive
is-IS-GudrunNeural|Female|General|Friendly, Positive
is-IS-GunnarNeural|Male|General|Friendly, Positive
it-IT-DiegoNeural|Male|General|Friendly, Positive
it-IT-ElsaNeural|Female|General|Friendly, Positive
it-IT-GiuseppeMultilingualNeural|Male|General|Friendly, Positive
it-IT-IsabellaNeural|Female|General|Friendly, Positive
iu-Cans-CA-SiqiniqNeural|Female|General|Friendly, Positive
iu-Cans-CA-TaqqiqNeural|Male|General|Friendly, Positive
iu-Latn-CA-SiqiniqNeural|Female|General|Friendly, Positive
iu-Latn-CA-TaqqiqNeural|Male|General|Friendly, Positive
ja-JP-KeitaNeural|Male|General|Friendly, Positive
ja-JP-NanamiNeural|Female|General|Friendly, Positive
jv-ID-DimasNeural|Male|General|Friendly, Positive
jv-ID-SitiNeural|Female|General|Friendly, Positive
ka-GE-EkaNeural|Female|General|Friendly, Positive
ka-GE-GiorgiNeural|Male|General|Friendly, Positive
kk-KZ-AigulNeural|Female|General|Friendly, Positive
kk-KZ-DauletNeural|Male|General|Friendly, Positive
km-KH-PisethNeural|Male|General|Friendly, Positive
km-KH-SreymomNeural|Female|General|Friendly, Positive
kn-IN-GaganNeural|Male|General|Friendly, Positive
kn-IN-SapnaNeural|Female|General|Friendly, Positive
ko-KR-HyunsuMultilingualNeural|Male|General|Friendly, Positive
ko-KR-InJoonNeural|Male|General|Friendly, Positive
ko-KR-SunHiNeural|Female|General|Friendly, Positive
lo-LA-ChanthavongNeural|Male|General|Friendly, Positive
lo-LA-KeomanyNeural|Female|General|Friendly, Positive
lt-LT-LeonasNeural|Male|General|Friendly, Positive
lt-LT-OnaNeural|Female|General|Friendly, Positive
lv-LV-EveritaNeural|Female|General|Friendly, Positive
lv-LV-NilsNeural|Male|General|Friendly, Positive
mk-MK-AleksandarNeural|Male|General|Friendly, Positive
mk-MK-MarijaNeural|Female|General|Friendly, Positive
ml-IN-MidhunNeural|Male|General|Friendly, Positive
ml-IN-SobhanaNeural|Female|General|Friendly, Positive
mn-MN-BataaNeural|Male|General|Friendly, Positive
mn-MN-YesuiNeural|Female|General|Friendly, Positive
mr-IN-AarohiNeural|Female|General|Friendly, Positive
mr-IN-ManoharNeural|Male|General|Friendly, Positive
ms-MY-OsmanNeural|Male|General|Friendly, Positive
ms-MY-YasminNeural|Female|General|Friendly, Positive
mt-MT-GraceNeural|Female|General|Friendly, Positive
mt-MT-JosephNeural|Male|General|Friendly, Positive
my-MM-NilarNeural|Female|General|Friendly, Positive
my-MM-ThihaNeural|Male|General|Friendly, Positive
nb-NO-FinnNeural|Male|General|Friendly, Positive
nb-NO-PernilleNeural|Female|General|Friendly, Positive
ne-NP-HemkalaNeural|Female|General|Friendly, Positive
ne-NP-SagarNeural|Male|General|Friendly, Positive
nl-BE-ArnaudNeural|Male|General|Friendly, Positive
nl-BE-DenaNeural|Female|General|Friendly, Positive
nl-NL-ColetteNeural|Female|General|Friendly, Positive
nl-NL-FennaNeural|Female|General|Friendly, Positive
nl-NL-MaartenNeural|Male|General|Friendly, Positive
pl-PL-MarekNeural|Male|General|Friendly, Positive
pl-PL-ZofiaNeural|Female|General|Friendly, Positive
ps-AF-GulNawazNeural|Male|General|Friendly, Positive
ps-AF-LatifaNeural|Female|General|Friendly, Positive
pt-BR-AntonioNeural|Male|General|Friendly, Positive
pt-BR-FranciscaNeural|Female|General|Friendly, Positive
pt-BR-ThalitaMultilingualNeural|Female|General|Friendly, Positive
pt-PT-DuarteNeural|Male|General|Friendly, Positive
pt-PT-RaquelNeural|Female|General|Friendly, Positive
ro-RO-AlinaNeural|Female|General|Friendly, Positive
ro-RO-EmilNeural|Male|General|Friendly, Positive
ru-RU-DmitryNeural|Male|General|Friendly, Positive
ru-RU-SvetlanaNeural|Female|General|Friendly, Positive
si-LK-SameeraNeural|Male|General|Friendly, Positive
si-LK-ThiliniNeural|Female|General|Friendly, Positive
sk-SK-LukasNeural|Male|General|Friendly, Positive
sk-SK-ViktoriaNeural|Female|General|Friendly, Positive
sl-SI-PetraNeural|Female|General|Friendly, Positive
sl-SI-RokNeural|Male|General|Friendly, Positive
so-SO-MuuseNeural|Male|General|Friendly, Positive
so-SO-UbaxNeural|Female|General|Friendly, Positive
sq-AL-AnilaNeural|Female|General|Friendly, Positive
sq-AL-IlirNeural|Male|General|Friendly, Positive
sr-RS-NicholasNeural|Male|General|Friendly, Positive
sr-RS-SophieNeural|Female|General|Friendly, Positive
su-ID-JajangNeural|Male|General|Friendly, Positive
su-ID-TutiNeural|Female|General|Friendly, Positive
sv-SE-MattiasNeural|Male|General|Friendly, Positive
sv-SE-SofieNeural|Female|General|Friendly, Positive
sw-KE-RafikiNeural|Male|General|Friendly, Positive
sw-KE-ZuriNeural|Female|General|Friendly, Positive
sw-TZ-DaudiNeural|Male|General|Friendly, Positive
sw-TZ-RehemaNeural|Female|General|Friendly, Positive
ta-IN-PallaviNeural|Female|General|Friendly, Positive
ta-IN-ValluvarNeural|Male|General|Friendly, Positive
ta-LK-KumarNeural|Male|General|Friendly, Positive
ta-LK-SaranyaNeural|Female|General|Friendly, Positive
ta-MY-KaniNeural|Female|General|Friendly, Positive
ta-MY-SuryaNeural|Male|General|Friendly, Positive
ta-SG-AnbuNeural|Male|General|Friendly, Positive
ta-SG-VenbaNeural|Female|General|Friendly, Positive
te-IN-MohanNeural|Male|General|Friendly, Positive
te-IN-ShrutiNeural|Female|General|Friendly, Positive
th-TH-NiwatNeural|Male|General|Friendly, Positive
th-TH-PremwadeeNeural|Female|General|Friendly, Positive
tr-TR-AhmetNeural|Male|General|Friendly, Positive
tr-TR-EmelNeural|Female|General|Friendly, Positive
uk-UA-OstapNeural|Male|General|Friendly, Positive
uk-UA-PolinaNeural|Female|General|Friendly, Positive
ur-IN-GulNeural|Female|General|Friendly, Positive
ur-IN-SalmanNeural|Male|General|Friendly, Positive
ur-PK-AsadNeural|Male|General|Friendly, Positive
ur-PK-UzmaNeural|Female|General|Friendly, Positive
uz-UZ-MadinaNeural|Female|General|Friendly, Positive
uz-UZ-SardorNeural|Male|General|Friendly, Positive
vi-VN-HoaiMyNeural|Female|General|Friendly, Positive
vi-VN-NamMinhNeural|Male|General|Friendly, Positive
zh-CN-XiaoxiaoNeural|Female|News, Novel|Warm
zh-CN-XiaoyiNeural|Female|Cartoon, Novel|Lively
zh-CN-YunjianNeural|Male|Sports,|Novel
zh-CN-YunxiNeural|Male|Novel|Lively, Sunshine
zh-CN-YunxiaNeural|Male|Cartoon, Novel|Cute
zh-CN-YunyangNeural|Male|News|Professional, Reliable
zh-CN-liaoning-XiaobeiNeural|Female|Dialect|Humorous
zh-CN-shaanxi-XiaoniNeural|Female|Dialect|Bright
zh-HK-HiuGaaiNeural|Female|General|Friendly, Positive
zh-HK-HiuMaanNeural|Female|General|Friendly, Positive
zh-HK-WanLungNeural|Male|General|Friendly, Positive
zh-TW-HsiaoChenNeural|Female|General|Friendly, Positive
zh-TW-HsiaoYuNeural|Female|General|Friendly, Positive
zh-TW-YunJheNeural|Male|General|Friendly, Positive
zu-ZA-ThandoNeural|Female|General|Friendly, Positive
zu-ZA-ThembaNeural|Male|General|Friendly, Positive
"""

LANGUAGE_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "az": "Azerbaijani",
    "bg": "Bulgarian", "bn": "Bengali", "bs": "Bosnian", "ca": "Catalan",
    "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German",
    "el": "Greek", "en": "English", "es": "Spanish", "et": "Estonian",
    "fa": "Persian", "fi": "Finnish", "fil": "Filipino", "fr": "French",
    "ga": "Irish", "gl": "Galician", "gu": "Gujarati", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian",
    "is": "Icelandic", "it": "Italian", "iu": "Inuktitut", "ja": "Japanese",
    "jv": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer",
    "kn": "Kannada", "ko": "Korean", "lo": "Lao", "lt": "Lithuanian",
    "lv": "Latvian", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian",
    "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese",
    "nb": "Norwegian", "ne": "Nepali", "nl": "Dutch", "pl": "Polish",
    "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "so": "Somali",
    "sq": "Albanian", "sr": "Serbian", "su": "Sundanese", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "zh": "Chinese", "zu": "Zulu",
}

REGION_NAMES = {
    "AE": "United Arab Emirates", "AF": "Afghanistan", "AL": "Albania",
    "AR": "Argentina", "AT": "Austria", "AU": "Australia", "AZ": "Azerbaijan",
    "BA": "Bosnia and Herzegovina", "BD": "Bangladesh", "BE": "Belgium",
    "BG": "Bulgaria", "BH": "Bahrain", "BO": "Bolivia", "BR": "Brazil",
    "CA": "Canada", "CH": "Switzerland", "CL": "Chile", "CN": "China",
    "CO": "Colombia", "CR": "Costa Rica", "CU": "Cuba", "CZ": "Czech Republic",
    "DE": "Germany", "DK": "Denmark", "DO": "Dominican Republic", "DZ": "Algeria",
    "EC": "Ecuador", "EE": "Estonia", "EG": "Egypt", "ES": "Spain",
    "ET": "Ethiopia", "FI": "Finland", "GB": "United Kingdom", "GE": "Georgia",
    "GQ": "Equatorial Guinea", "GR": "Greece", "GT": "Guatemala", "HK": "Hong Kong",
    "HN": "Honduras", "HR": "Croatia", "HU": "Hungary", "ID": "Indonesia",
    "IE": "Ireland", "IL": "Israel", "IN": "India", "IQ": "Iraq", "IR": "Iran",
    "IS": "Iceland", "IT": "Italy", "JO": "Jordan", "JP": "Japan", "KE": "Kenya",
    "KH": "Cambodia", "KR": "South Korea", "KW": "Kuwait", "KZ": "Kazakhstan",
    "LA": "Laos", "LB": "Lebanon", "LK": "Sri Lanka", "LT": "Lithuania",
    "LV": "Latvia", "LY": "Libya", "MA": "Morocco", "MK": "North Macedonia",
    "MM": "Myanmar", "MN": "Mongolia", "MT": "Malta", "MX": "Mexico",
    "MY": "Malaysia", "NG": "Nigeria", "NI": "Nicaragua", "NL": "Netherlands",
    "NO": "Norway", "NP": "Nepal", "NZ": "New Zealand", "OM": "Oman",
    "PA": "Panama", "PE": "Peru", "PH": "Philippines", "PK": "Pakistan",
    "PL": "Poland", "PR": "Puerto Rico", "PT": "Portugal", "PY": "Paraguay",
    "QA": "Qatar", "RO": "Romania", "RS": "Serbia", "RU": "Russia",
    "SA": "Saudi Arabia", "SE": "Sweden", "SG": "Singapore", "SI": "Slovenia",
    "SK": "Slovakia", "SO": "Somalia", "SV": "El Salvador", "SY": "Syria",
    "TH": "Thailand", "TN": "Tunisia", "TR": "Turkey", "TW": "Taiwan",
    "TZ": "Tanzania", "UA": "Ukraine", "US": "United States", "UY": "Uruguay",
    "UZ": "Uzbekistan", "VE": "Venezuela", "VN": "Vietnam", "YE": "Yemen",
    "ZA": "South Africa",
}

SCRIPT_NAMES = {"Cans": "Canadian Aboriginal syllabics", "Latn": "Latin script"}


def pretty_words(value: str) -> str:
    value = re.sub(r"(Multilingual|Expressive)?Neural$", lambda m: (" " + (m.group(1) or "")), value)
    value = re.sub(r"(?<!^)(?=[A-Z])", " ", value)
    return " ".join(value.split())


def catalog_records() -> list[dict]:
    records = []
    for line in VOICE_CATALOG.strip().splitlines():
        voice_id, gender, categories, personality = line.split("|", 3)
        pieces = voice_id.split("-")
        language_code = pieces[0]
        region_code = next((x for x in pieces[1:-1] if x in REGION_NAMES), "")
        script_code = next((x for x in pieces[1:-1] if x in SCRIPT_NAMES), "")
        name = pretty_words(pieces[-1])
        language = LANGUAGE_NAMES.get(language_code, language_code)
        region = REGION_NAMES.get(region_code, region_code)
        script = SCRIPT_NAMES.get(script_code, "")
        base_selector = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        records.append({
            "id": voice_id, "name": name, "gender": gender,
            "categories": categories, "personality": personality,
            "language": language, "region": region, "script": script,
            "selector": base_selector,
        })

    counts = {}
    for record in records:
        counts[record["selector"]] = counts.get(record["selector"], 0) + 1
    used = set()
    for record in records:
        selector = record["selector"]
        if counts[selector] > 1:
            place = record["script"] or record["region"] or record["language"]
            place = re.sub(r"[^a-z0-9]+", "-", place.lower()).strip("-")
            selector = f"{selector}-{place}"
        if selector in used:
            selector += "-voice"
        record["selector"] = selector
        used.add(selector)
    return records


BASE_VOICES = catalog_records()

# Edge currently provides only two native Hindi models (Madhur and Swara).
# These aliases expose Microsoft's strongest multilingual voices as additional
# Hindi/Hinglish choices. They are genuine voice choices, but are clearly
# labelled multilingual instead of being misrepresented as native Hindi.
HINDI_MULTILINGUAL_ALIASES = {
    "brian-hindi": "brian-multilingual",
    "andrew-hindi": "andrew-multilingual",
    "ava-hindi": "ava-multilingual",
    "emma-hindi": "emma-multilingual",
}

_base_by_selector = {v["selector"]: v for v in BASE_VOICES}
HINDI_MULTILINGUAL_VOICES = []
for alias, source_selector in HINDI_MULTILINGUAL_ALIASES.items():
    source = _base_by_selector[source_selector]
    alternative = source.copy()
    alternative.update({
        "name": f"{source['name']} — Hindi/Hinglish",
        "language": "Hindi",
        "region": "Multilingual alternatives",
        "script": "",
        "selector": alias,
        "categories": "Multilingual Hindi/Hinglish alternative",
    })
    HINDI_MULTILINGUAL_VOICES.append(alternative)

ALL_VOICES = BASE_VOICES + HINDI_MULTILINGUAL_VOICES
VOICE_BY_SELECTOR = {v["selector"]: v for v in ALL_VOICES}
# Exact Microsoft IDs resolve to their original catalog entries.
VOICE_BY_ID = {v["id"].lower(): v for v in BASE_VOICES}
UNIQUE_VOICE_MODELS = len({v["id"] for v in BASE_VOICES})

# Styles are deliberately subtle. Large rate/pitch changes make Edge voices
# sound synthetic, so natural prosody is prioritized over exaggerated effects.
STYLES = {
    "natural": (0, 0, "Unmodified native voice delivery"),
    "narrative": (-4, 0, "Slightly slower storytelling"),
    "deep": (-6, 0, "Slower delivery with gentle low-end warmth"),
    "emotional": (-3, 0, "Natural pacing with open dynamics"),
    "warm": (-4, 0, "Soft, relaxed and intimate"),
    "cinematic": (-7, 0, "Measured documentary pacing"),
}

DEFAULT_VOICE = "brian-multilingual"
SAMPLE_RATE = 48000
MAX_CHARS = 4800


def voice_base_settings(voice: dict) -> tuple[str, str]:
    """Keep Microsoft's native voice character and prosody unchanged."""
    return "+0%", "+0Hz"


def ensure_package(package: str, import_name: str | None = None) -> None:
    try:
        __import__(import_name or package.replace("-", "_"))
    except ImportError:
        print(f"Installing {package}...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def runtime_family() -> str:
    """Identify the runtime silently; nothing is printed to the terminal."""
    prefix = os.environ.get("PREFIX", "").lower()
    if os.name == "nt":
        return "windows"
    if "com.termux" in prefix or "termux" in prefix:
        return "termux"
    if Path("/etc/debian_version").exists():
        return "debian"
    return "linux"


def ffmpeg_bin() -> str:
    family = runtime_family()
    executable = "ffmpeg.exe" if family == "windows" else "ffmpeg"

    # Support FFmpeg beside the script, on PATH, or supplied by imageio-ffmpeg.
    local_candidates = (
        Path(__file__).resolve().parent / executable,
        Path(sys.executable).resolve().parent / executable,
    )
    for candidate in local_candidates:
        if candidate.is_file():
            return str(candidate)

    native = shutil.which(executable) or shutil.which("ffmpeg")
    if native:
        return native

    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).is_file():
            return bundled
    except (ImportError, RuntimeError, OSError):
        pass

    raise RuntimeError(
        "FFmpeg was not found. Install FFmpeg for your system or install "
        "the imageio-ffmpeg Python package."
    )


def run_ffmpeg(args: list[str]) -> None:
    options = {"check": True}
    if runtime_family() == "windows":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run(
        [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", *args],
        **options,
    )


def clean_text(text: str) -> str:
    text = html.unescape(text.replace("&nbsp;", " "))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?।])", r"\1", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def split_naturally(text: str, style: str = "natural") -> list[str]:
    """Create long coherent passages instead of resetting at every paragraph."""
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        # Keep adjacent paragraphs in one Edge request whenever possible. This
        # preserves the model's tone, cadence, and vocal identity.
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= MAX_CHARS:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= MAX_CHARS:
            current = paragraph
            continue

        # Only unusually long paragraphs are split at sentence boundaries.
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?।])\s+", paragraph)
            if sentence.strip()
        ]
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > MAX_CHARS:
                chunks.append(current)
                current = sentence
            else:
                current = candidate

    if current:
        chunks.append(current)
    return chunks


def setting_value(base: str, offset: int, unit: str) -> str:
    value = int(re.search(r"[-+]?\d+", base).group()) + offset
    sign = "+" if value >= 0 else ""
    return f"{sign}{value}{unit}"


async def synthesize_chunk(
    text: str, voice_id: str, rate: str, pitch: str, destination: Path
) -> None:
    import edge_tts

    errors = []
    settings = [(rate, pitch), ("-2%", "+0Hz"), ("+0%", "+0Hz")]
    for attempt, (use_rate, use_pitch) in enumerate(settings, start=1):
        try:
            communicator = edge_tts.Communicate(
                text=text,
                voice=voice_id,
                rate=use_rate,
                pitch=use_pitch,
                volume="+0%",
            )
            await communicator.save(str(destination))
            if destination.exists() and destination.stat().st_size > 1000:
                return
            raise RuntimeError("the generated audio file was empty")
        except Exception as exc:
            errors.append(str(exc))
            if destination.exists():
                destination.unlink()
            if attempt < len(settings):
                await asyncio.sleep(0.8 * attempt)

    raise RuntimeError(f"TTS generation failed: {errors[-1]}")


def make_silence(milliseconds: int, destination: Path) -> None:
    run_ffmpeg([
        "-f", "lavfi",
        "-t", f"{milliseconds / 1000:.3f}",
        "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
        "-c:a", "pcm_s24le",
        str(destination),
    ])


def convert_to_wav(source: Path, destination: Path) -> None:
    run_ffmpeg([
        "-i", str(source),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s24le",
        str(destination),
    ])


def concat_audio(parts: list[Path], destination: Path) -> None:
    list_file = destination.parent / "concat.txt"
    with list_file.open("w", encoding="utf-8") as handle:
        for part in parts:
            escaped = part.as_posix().replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")

    run_ffmpeg([
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s24le",
        str(destination),
    ])


def master_audio(source: Path, destination: Path, style: str) -> None:
    """Apply gentle mastering without altering the neural voice's pitch."""
    # No asetrate, pitch shifter, crusher, room noise, or aggressive de-esser.
    # Those effects can make neural speech metallic or robotic.
    profiles = {
        "natural": (
            "highpass=f=55,"
            "acompressor=threshold=-16dB:ratio=1.35:attack=25:release=220:makeup=0.5dB,"
        ),
        "narrative": (
            "highpass=f=55,"
            "equalizer=f=180:t=q:w=0.9:g=0.7,"
            "equalizer=f=3400:t=q:w=1.2:g=0.8,"
            "acompressor=threshold=-18dB:ratio=1.55:attack=25:release=230:makeup=0.8dB,"
        ),
        "deep": (
            "highpass=f=48,"
            "equalizer=f=110:t=q:w=0.8:g=1.8,"
            "equalizer=f=260:t=q:w=1.0:g=-0.7,"
            "equalizer=f=3000:t=q:w=1.2:g=0.6,"
            "acompressor=threshold=-19dB:ratio=1.8:attack=28:release=250:makeup=1dB,"
        ),
        "emotional": (
            "highpass=f=55,"
            "equalizer=f=3800:t=q:w=1.1:g=1.0,"
            "acompressor=threshold=-13dB:ratio=1.25:attack=35:release=280:makeup=0.3dB,"
        ),
        "warm": (
            "highpass=f=50,"
            "equalizer=f=140:t=q:w=0.9:g=1.2,"
            "equalizer=f=6500:t=q:w=1.0:g=-0.6,"
            "acompressor=threshold=-17dB:ratio=1.45:attack=30:release=260:makeup=0.6dB,"
        ),
        "cinematic": (
            "highpass=f=48,"
            "equalizer=f=100:t=q:w=0.8:g=1.5,"
            "equalizer=f=2800:t=q:w=1.1:g=1.0,"
            "acompressor=threshold=-20dB:ratio=2.0:attack=32:release=280:makeup=1.2dB,"
        ),
    }
    filters = profiles[style] + "loudnorm=I=-16:TP=-1.5:LRA=11"

    if destination.suffix.lower() == ".mp3":
        codec = ["-c:a", "libmp3lame", "-b:a", "320k"]
    else:
        codec = ["-c:a", "pcm_s24le", "-ar", str(SAMPLE_RATE)]

    run_ffmpeg([
        "-i", str(source),
        "-af", filters,
        "-ac", "1",
        *codec,
        str(destination),
    ])


async def build_voiceover(
    chunks: list[str], voice_id: str, rate: str, pitch: str,
    workdir: Path, style: str
) -> Path:
    parts: list[Path] = []
    pause_lengths = {
        "natural": 480,
        "narrative": 620,
        "deep": 680,
        "emotional": 240,
        "warm": 600,
        "cinematic": 780,
    }
    pause = workdir / "style_pause.wav"
    make_silence(pause_lengths[style], pause)

    for index, chunk in enumerate(chunks, start=1):
        encoded = workdir / f"speech_{index:04d}.mp3"
        wav = workdir / f"speech_{index:04d}.wav"

        # Keep one stable setting throughout each passage. Artificial pitch
        # cycling between sentences causes audible voice resets and artifacts.
        chunk_rate = rate
        chunk_pitch = pitch
        await synthesize_chunk(
            chunk, voice_id, chunk_rate, chunk_pitch, encoded
        )
        convert_to_wav(encoded, wav)
        parts.append(wav)
        if index < len(chunks):
            parts.append(pause)
        print(
            f"  Segment {index}/{len(chunks)} "
            f"(rate {chunk_rate}, pitch {chunk_pitch})",
            file=sys.stderr,
        )
        await asyncio.sleep(0.08)

    joined = workdir / "joined.wav"
    concat_audio(parts, joined)
    return joined


def narration_recommended(voice: dict) -> bool:
    curated = {
        "brian", "brian-multilingual", "brian-hindi",
        "andrew", "andrew-multilingual", "andrew-hindi",
        "christopher", "guy", "aria", "ava", "ava-multilingual",
        "ava-hindi", "emma", "emma-multilingual", "emma-hindi",
        "madhur", "swara", "ryan", "thomas", "sonia",
    }
    if voice["selector"] in curated:
        return True
    text = f'{voice["categories"]} {voice["personality"]}'.lower()
    keywords = (
        "news", "novel", "warm", "confident", "authority", "passion",
        "reliable", "professional", "expressive", "authentic", "sincere",
    )
    return any(keyword in text for keyword in keywords)


def list_voices(query: str | None = None) -> None:
    voices = ALL_VOICES
    if query:
        needle = query.casefold()
        if needle in {"recommended", "narration", "mature"}:
            voices = [voice for voice in voices if narration_recommended(voice)]
        else:
            voices = [
                voice for voice in voices
                if needle in " ".join((
                    voice["language"], voice["region"], voice["name"],
                    voice["gender"], voice["personality"], voice["selector"],
                )).casefold()
            ]

    result_models = len({voice["id"] for voice in voices})
    print(
        f"\nAvailable choices: {len(voices)} "
        f"({result_models} voice models in this result; "
        f"{UNIQUE_VOICE_MODELS} Microsoft models total)"
    )
    print("★ = especially suitable for narration based on Microsoft's labels")
    print("Hindi includes 2 native voices plus clearly labelled multilingual alternatives.")
    print("Use the selector shown in brackets; locale abbreviations are hidden.\n")

    languages = sorted({voice["language"] for voice in voices})
    for language in languages:
        print(f"========== {language.upper()} ==========")
        language_voices = [v for v in voices if v["language"] == language]
        regions = sorted({v["region"] or "Default" for v in language_voices})
        for region in regions:
            print(f"  {region}")
            region_voices = [
                v for v in language_voices
                if (v["region"] or "Default") == region
            ]
            for voice in sorted(region_voices, key=lambda item: item["name"]):
                star = "★" if narration_recommended(voice) else " "
                script = f"; {voice['script']}" if voice["script"] else ""
                default = " [default]" if voice["selector"] == DEFAULT_VOICE else ""
                print(
                    f"    {star} {voice['name']:<24} {voice['gender']:<6} "
                    f"[{voice['selector']}]{default}{script}"
                )
                print(f"       {voice['personality']} — {voice['categories']}")
        print()

    if not voices:
        print(f"No voices matched: {query}\n")

    print("Styles:")
    for name, (_, _, description) in STYLES.items():
        print(f"  {name:<12} {description}")
    print("\nExamples:")
    print("  python app.py list English")
    print("  python app.py list Hindi")
    print("  python app.py list Spanish")
    print("  python app.py brian-multilingual script.txt output.mp3 --style narrative")
    print("  python app.py madhur hindi.txt hindi.mp3 --style natural")
    print("  python app.py brian-hindi hinglish.txt hinglish.mp3 --style natural\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViralTTS neural voiceover generator")
    parser.add_argument("voice", nargs="?", help="Voice name, or 'list'")
    parser.add_argument("script", nargs="?", help="UTF-8 text file")
    parser.add_argument("output", nargs="?", default="voiceover.wav", help="Output WAV or MP3")
    parser.add_argument("--style", choices=STYLES, default="narrative", help="Delivery style")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not args.voice or args.voice.casefold() == "list":
        list_voices(args.script)
        return

    voice = VOICE_BY_SELECTOR.get(args.voice.casefold())
    if voice is None:
        voice = VOICE_BY_ID.get(args.voice.casefold())
    if voice is None:
        print(f"Unknown voice selector: {args.voice}", file=sys.stderr)
        print("Run 'python app.py list' or 'python app.py list English'.", file=sys.stderr)
        raise SystemExit(1)

    if not args.script:
        print("Missing script file.", file=sys.stderr)
        print("Usage: python app.py <voice> script.txt [output.wav|output.mp3] --style narrative")
        raise SystemExit(1)

    script_path = Path(args.script).expanduser()
    if not script_path.is_file():
        print(f"Script file not found: {script_path}", file=sys.stderr)
        raise SystemExit(1)

    output = Path(args.output).expanduser()
    if output.suffix.lower() not in {".wav", ".mp3"}:
        output = output.with_suffix(".wav")
    output.parent.mkdir(parents=True, exist_ok=True)

    ensure_package("edge-tts", "edge_tts")
    ffmpeg_bin()  # Check before spending time synthesizing.

    voice_id = voice["id"]
    base_rate, base_pitch = voice_base_settings(voice)
    rate_offset, pitch_offset, _ = STYLES[args.style]
    rate = setting_value(base_rate, rate_offset, "%")
    pitch = setting_value(base_pitch, pitch_offset, "Hz")

    chunks = split_naturally(
        script_path.read_text(encoding="utf-8-sig"), args.style
    )
    if not chunks:
        print("The script file is empty.", file=sys.stderr)
        raise SystemExit(1)

    print(
        f"Voice: {voice['name']} ({voice['gender']}) — {voice['personality']}",
        file=sys.stderr,
    )
    print(
        f"Language: {voice['language']} — {voice['region']}; "
        f"style: {args.style}; rate: {rate}; pitch: {pitch}",
        file=sys.stderr,
    )
    print(f"Passages: {len(chunks)}", file=sys.stderr)

    try:
        with tempfile.TemporaryDirectory(prefix="viraltts_") as temp_dir:
            workdir = Path(temp_dir)
            joined = asyncio.run(
                build_voiceover(
                    chunks, voice_id, rate, pitch, workdir, args.style
                )
            )
            print("Mastering clean 48 kHz audio...", file=sys.stderr)
            master_audio(joined, output, args.style)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130)
    except subprocess.CalledProcessError as exc:
        print(f"FFmpeg failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Done: {output} ({size_mb:.2f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
