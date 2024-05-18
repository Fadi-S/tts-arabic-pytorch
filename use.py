
from models.tacotron2 import Tacotron2Wave
import soundfile as sf  # for saving audio files


# model = Tacotron2Wave("pretrained/tacotron2_ar_adv.pth")
model = Tacotron2Wave("states_7000.pth")
# model = model.cuda()

import text

lines = [
    "اهلا بيك في حلقة جديدة من برنامج الذكاء الاقتصادي",
    "ازيك عامل ايه يا باشا",
    "مساء الورد و الياسمين علي عنيكي الحلوين",
    "اهلا و سهلا",
]

for i in range(len(lines)):
    line = lines[i]
    wave, mel_spec = model.tts(text.arabic_to_buckwalter(line), return_mel=True, denoise=0.005)

    output_file_path = f"output_{i}.wav"
    sf.write(output_file_path, wave, 22050)
