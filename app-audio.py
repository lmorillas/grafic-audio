import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import io

# Título de la aplicación
st.title("Análisis de Audio - Gráficos Comparativos")

# Subir múltiples archivos de audio
uploaded_files = st.sidebar.file_uploader("Sube uno o más archivos de audio", type=['wav', 'mp3'], accept_multiple_files=True)

# Slider para seleccionar el rango máximo de frecuencias a mostrar
freq_range_max = st.sidebar.slider(
    "Selecciona el rango máximo de frecuencias a mostrar (Hz)", 
    min_value=1000, 
    max_value=20000, 
    value=10000, 
    step=500
)

# Checkbox para seleccionar qué gráficos mostrar
show_waveplot = st.sidebar.checkbox("Mostrar Gráfica de Onda (Waveplot)", True)
show_spectrogram = st.sidebar.checkbox("Mostrar Espectrograma", True)
show_chroma = st.sidebar.checkbox("Mostrar Chroma", True)
show_mfcc = st.sidebar.checkbox("Mostrar MFCC", True)
show_frequency_spectrum = st.sidebar.checkbox("Mostrar Espectro de Frecuencias", True)

# Función para generar el espectrograma
def plot_spectrogram(audio_data, sr, ax):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title('Espectrograma')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Frecuencia (Hz)')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')

# Función para graficar el Chroma
def plot_chroma(audio_data, sr, ax):
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    ax.set_title('Chroma')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Pitch Classes')
    plt.colorbar(img, ax=ax)

# Función para graficar el MFCC
def plot_mfcc(audio_data, sr, ax):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    ax.set_title('MFCC')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('MFCC Coefficients')
    plt.colorbar(img, ax=ax)

# Función para graficar el Espectro de Frecuencias
def plot_frequency_spectrum(audio_data, sr, ax, freq_range_max):
    # Calcular la transformada de Fourier
    fft = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(fft))

    # Extraer el espectro de magnitudes
    magnitude = np.abs(fft)

    # Solo tomamos la mitad del espectro ya que es simétrico
    left_spectrum = magnitude[:len(magnitude)//2]
    left_freqs = freqs[:len(freqs)//2] * sr

    # Filtrar el espectro a las frecuencias menores al valor máximo seleccionado
    mask = left_freqs <= freq_range_max
    filtered_spectrum = left_spectrum[mask]
    filtered_freqs = left_freqs[mask]

    # Graficar el espectro de frecuencias
    ax.plot(filtered_freqs, filtered_spectrum)
    ax.set_title('Espectro de Frecuencias')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Amplitud')

# Inicializar la lista de gráficos y datos
if uploaded_files:
    st.header("Análisis Comparativo de Audios")

    # Crear subplots dinámicamente en función de cuántos audios se suban
    for uploaded_file in uploaded_files:
        audio_data, sr = librosa.load(uploaded_file, sr=None)

        st.subheader(f"Archivo: {uploaded_file.name}")

        # Crear subplots para cada archivo de audio
        if show_waveplot:
            # Gráfica de la onda del audio
            st.write(f"Gráfica de Onda (Waveplot) - {uploaded_file.name}")
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(audio_data, sr=sr, ax=ax)
            ax.set_title(f"Onda del Audio - {uploaded_file.name}")
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            st.pyplot(fig)

        if show_spectrogram:
            # Espectrograma
            st.write(f"Espectrograma - {uploaded_file.name}")
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_spectrogram(audio_data, sr, ax)
            st.pyplot(fig)

        if show_chroma:
            # Chroma
            st.write(f"Chroma - {uploaded_file.name}")
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_chroma(audio_data, sr, ax)
            st.pyplot(fig)

        if show_mfcc:
            # MFCC
            st.write(f"MFCC - {uploaded_file.name}")
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_mfcc(audio_data, sr, ax)
            st.pyplot(fig)

        if show_frequency_spectrum:
            # Espectro de Frecuencias
            st.write(f"Espectro de Frecuencias - {uploaded_file.name}")
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_frequency_spectrum(audio_data, sr, ax, freq_range_max)
            st.pyplot(fig)

else:
    st.write("Sube uno o más archivos de audio para analizarlos.")
