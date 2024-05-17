
from models.tacotron2 import Tacotron2Wave
import soundfile as sf  # for saving audio files


# model = Tacotron2Wave("pretrained/tacotron2_ar_adv.pth")
model = Tacotron2Wave("states.pth")
# model = model.cuda()

import text

line = "اهلا بيك في حلقة جديدة من برنامج الذكاء الصناعي"
line2 = "ازيك عامل ايه يا باشا"
line3 = "مساء الورد و الياسمين علي عنيكي الحلوين"

wave, mel_spec = model.tts(text.arabic_to_buckwalter(line3), return_mel=True, denoise=0.005)

output_file_path = "output.wav"
sf.write(output_file_path, wave, 22050)
