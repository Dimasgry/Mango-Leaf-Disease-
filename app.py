from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model
model = load_model('Daunmangga.keras')

# Nama kelas
class_names = [


    'Anthracnose',
    'Bacterial Canker',
    'Cutting Weevil',
    'Die Back',
    'Gall Midge',
    'Healthy',
    'Powdery Mildew',
    'Sooty Mould'
]

disease_info = {

    "Anthracnose": {
        "nama": "Antraknosa",
        "deskripsi": "Antraknosa merupakan penyakit jamur yang umum menyerang daun mangga, ditandai dengan munculnya bercak cokelat kehitaman berbentuk tidak beraturan. Bercak dapat melebar dan menyebabkan jaringan daun mengering serta gugur.",
        "penyebab": "Disebabkan oleh jamur Colletotrichum spp. yang berkembang pada kondisi lembap dan curah hujan tinggi.",
        "solusi": "Lakukan pemangkasan bagian yang terinfeksi, menjaga sirkulasi udara tanaman, dan menggunakan fungisida sesuai dosis yang dianjurkan."
    },

    "Bacterial Canker": {
        "nama": "Kanker Bakteri",
        "deskripsi": "Penyakit yang ditandai bercak gelap pada daun yang sering dikelilingi warna kekuningan. Pada serangan berat daun dapat mengering dan rontok.",
        "penyebab": "Disebabkan oleh infeksi bakteri patogen yang menyerang jaringan daun dan batang.",
        "solusi": "Buang bagian tanaman yang terinfeksi dan lakukan sanitasi kebun secara rutin untuk mencegah penyebaran bakteri."
    },

    "Cutting Weevil": {
        "nama": "Kumbang Pemotong Daun",
        "deskripsi": "Hama serangga yang memakan jaringan daun sehingga tepi daun terlihat terpotong atau berlubang.",
        "penyebab": "Serangan hama kumbang pemotong daun yang aktif memakan jaringan daun.",
        "solusi": "Lakukan pengendalian hama secara mekanis atau menggunakan insektisida yang sesuai."
    },

    "Die Back": {
        "nama": "Penyakit Mati Pucuk",
        "deskripsi": "Penyakit yang menyebabkan daun dan pucuk mengering secara bertahap hingga mati.",
        "penyebab": "Umumnya berkaitan dengan infeksi jamur serta kondisi tanaman yang lemah.",
        "solusi": "Pangkas bagian yang terserang dan tingkatkan kesehatan tanaman melalui pemupukan yang tepat."
    },

    "Gall Midge": {
        "nama": "Lalat Puru Daun",
        "deskripsi": "Hama yang menyebabkan terbentuknya puru atau benjolan pada daun sehingga pertumbuhannya tidak normal.",
        "penyebab": "Larva Gall Midge berkembang di jaringan daun dan memicu pembentukan puru.",
        "solusi": "Buang daun yang terserang dan lakukan pengendalian hama secara rutin."
    },

    "Healthy": {
        "nama": "Daun Sehat",
        "deskripsi": "Daun berwarna hijau segar, permukaan halus, dan tidak menunjukkan gejala serangan hama maupun penyakit.",
        "penyebab": "Tanaman berada dalam kondisi sehat dengan pertumbuhan yang optimal.",
        "solusi": "Pertahankan perawatan tanaman yang baik, termasuk penyiraman dan pemupukan yang teratur."
    },

    "Powdery Mildew": {
        "nama": "Embun Tepung",
        "deskripsi": "Penyakit jamur yang ditandai lapisan putih seperti tepung pada permukaan daun.",
        "penyebab": "Disebabkan oleh infeksi jamur embun tepung yang berkembang pada kondisi lingkungan tertentu.",
        "solusi": "Kurangi kelembapan berlebih dan gunakan fungisida yang sesuai bila diperlukan."
    },

    "Sooty Mould": {
        "nama": "Jamur Jelaga",
        "deskripsi": "Ditandai dengan lapisan hitam menyerupai jelaga pada permukaan daun sehingga menghambat fotosintesis.",
        "penyebab": "Tumbuh akibat embun madu yang dihasilkan oleh hama seperti kutu daun, kutu putih, atau kutu sisik.",
        "solusi": "Kendalikan hama penghasil embun madu dan bersihkan bagian tanaman yang terinfeksi."
    }
}
UPLOAD_FOLDER = os.path.join('static', 'uploads')

# otomatis membuat folder uploads
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi preprocessing gambar

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = prepare_image(filepath)

        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)
        
        info = disease_info[predicted_class]

        return render_template(
    'result.html',
    prediction=predicted_class,
    confidence=round(confidence, 2),
    image_path=filepath,

    disease_name=info['nama'],
    disease_desc=info['deskripsi'],
    disease_cause=info['penyebab'],
    disease_solution=info['solusi']
)

if __name__ == '__main__':
    app.run(debug=True)
    
    
    UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)