use std::error::Error;
use std::path::Path;

use hound::{WavReader, SampleFormat};
use rustfft::{FftPlanner, num_complex::Complex};
use plotters::prelude::*;
use minifb::{Window, WindowOptions, Key};

/// Читает WAV-файл и возвращает моно‑сэмплы (f32) и частоту дискретизации.
/// Поддерживает только 16‑битные PCM файлы.
fn read_wav<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    if spec.sample_format != SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err("Поддерживаются только 16‑битные PCM файлы".into());
    }

    let samples: Vec<i16> = reader.samples::<i16>()
        .map(|s| s.unwrap())
        .collect();

    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let mono: Vec<f32> = if channels == 1 {
        samples.iter().map(|&s| s as f32 / 32768.0).collect()
    } else {
        samples.chunks_exact(channels)
            .map(|chunk| {
                let sum: i32 = chunk.iter().map(|&s| s as i32).sum();
                (sum as f32) / (channels as f32 * 32768.0)
            })
            .collect()
    };

    Ok((mono, sample_rate))
}

/// Вычисляет односторонний амплитудный спектр (в децибелах) для моно‑сигнала.
/// Возвращает векторы частот (Гц) и амплитуд (дБ).
fn compute_spectrum(samples: &[f32], sample_rate: u32) -> (Vec<f64>, Vec<f64>) {
    let n = samples.len();
    let fft_size = n.next_power_of_two();
    let mut buffer: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); fft_size];
    for (i, &s) in samples.iter().enumerate() {
        buffer[i] = Complex::new(s as f64, 0.0);
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut buffer);

    let half = fft_size / 2;
    let mut freqs = Vec::with_capacity(half + 1);
    let mut mags = Vec::with_capacity(half + 1);

    for i in 0..=half {
        let c = buffer[i];
        let magnitude = c.norm();
        let db = 20.0 * (magnitude + 1e-12).log10();
        let freq = i as f64 * sample_rate as f64 / fft_size as f64;
        freqs.push(freq);
        mags.push(db);
    }

    (freqs, mags)
}

/// Рисует спектрограмму в буфер RGB и возвращает его.
fn plot_spectrum(
    freqs: &[f64],
    mags: &[f64],
    width: usize,
    height: usize,
    title: &str,
) -> Vec<u8> {
    let mut buf = vec![0; width * height * 3];

    {
        let root = BitMapBackend::with_buffer(&mut buf, (width as u32, height as u32))
            .into_drawing_area();
        root.fill(&WHITE).unwrap();

        let max_freq = freqs.last().unwrap_or(&0.0).ceil() as f32;
        let min_mag = mags.iter().fold(f64::INFINITY, |a, &b| a.min(b)) as f32;
        let max_mag = mags.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) as f32;

        let mag_padding = (max_mag - min_mag) * 0.05;
        let y_min = min_mag - mag_padding;
        let y_max = max_mag + mag_padding;

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..max_freq, y_min..y_max)
            .unwrap();

        chart.configure_mesh()
            .x_desc("Частота (Гц)")
            .y_desc("Амплитуда (дБ)")
            .draw()
            .unwrap();

        chart.draw_series(LineSeries::new(
            freqs.iter().zip(mags.iter()).map(|(&f, &m)| (f as f32, m as f32)),
            &RED,
        )).unwrap()
            .label("Спектр")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();
    }

    buf
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let filename = args.get(1).cloned().unwrap_or_else(|| "test_800hz.wav".to_string());

    println!("Чтение файла: {}", filename);

    let (samples, sample_rate) = read_wav(&filename)?;
    println!("Файл загружен: {} сэмплов, частота {} Гц", samples.len(), sample_rate);

    let (freqs, mags) = compute_spectrum(&samples, sample_rate);
    println!("Спектр вычислен. Диапазон частот: 0 .. {:.1} Гц", freqs.last().unwrap());

    let width = 900;
    let height = 600;
    let title = format!("Спектр файла: {}", filename);
    let rgb_buf = plot_spectrum(&freqs, &mags, width, height, &title);

    // Конвертируем RGB → ARGB (u32) для minifb
    let fb: Vec<u32> = rgb_buf
        .chunks_exact(3)
        .map(|rgb| 0xFF000000 | ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | (rgb[2] as u32))
        .collect();

    let mut window = Window::new(
        &title,
        width,
        height,
        WindowOptions::default(),
    )?;

    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        window.update_with_buffer(&fb, width, height)?;
    }

    Ok(())
}
