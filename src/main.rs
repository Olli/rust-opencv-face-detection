use std::thread;
use std::time::Duration;

use opencv::core::{Mat, MatTrait, Point, Rect, Rect_, Size, TickMeter, VecN, Vector};

use opencv::highgui::QT_FONT_NORMAL;
use opencv::objdetect::FaceDetectorYN;
// #[allow(unused_imports)]
use opencv::prelude::*;

#[allow(unused_imports)]
use opencv::{Result, core, highgui, imgproc, imgproc::*, videoio, videoio::*};

opencv::opencv_branch_5! {
    use opencv::xobjdetect::{CascadeClassifier, CASCADE_SCALE_IMAGE};
}

opencv::not_opencv_branch_5! {
    #[allow(unused_imports)]
    use opencv::objdetect::{CascadeClassifier, CASCADE_SCALE_IMAGE};
}

// fn visualize(frame: &mut Mat, faces: &mut Vector<Rect_<i32>>, fps: i32)  -> Result<(), Box<dyn std::error::Error>>{
fn visualize(frame: &mut Mat, faces: &mut Mat, fps: i32) -> Result<(), Box<dyn std::error::Error>> {
    let box_color = (0, 255, 0).into();
    let landmark_color: Vec<VecN<f64, 4>> = vec![
        (255, 0, 0).into(),   // right eye
        (0, 0, 255).into(),   // left eye
        (0, 255, 0).into(),   // nose tip
        (255, 0, 255).into(), // right mouth corner
        (0, 255, 255).into(), // left mouth corner
    ];
    let text_color = (0, 255, 0).into();
    if fps >= 0 {
        imgproc::put_text_def(
            frame,
            format!("FPS: {}", fps).as_str(),
            Point::new(0, 15),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
        )
        .unwrap();
    }
    let face_rows = faces.rows();
    for i in 0..face_rows {
        // get the position of the face
        let x1 = *faces.at_2d::<f32>(i, 0)? as i32;
        let y1 = *faces.at_2d::<f32>(i, 1)? as i32;
        let w = *faces.at_2d::<f32>(i, 2)? as i32;
        let h = *faces.at_2d::<f32>(i, 3)? as i32;

        // get and paint the confidence level
        let confidence: f32 = *faces.at_2d::<f32>(i, 14)?;
        imgproc::put_text(
            frame,
            format!("Confidence: {}", confidence).as_str(),
            Point::new(x1, y1 - 20),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            LINE_AA,
            false,
        )?;

        // imgproc::rectangle(frame, Rect::new(x1, y1, w, h), box_color, -1, 0, 0)?;
        // green rectangle around a detected face
        imgproc::rectangle(frame, Rect::new(x1, y1, w, h), box_color, 1, 0, 0)?;

        // let ksize = Size::new(3, 3); // instead of 3
        // let tf = frame.clone();
        // imgproc::box_filter_def(&tf,  frame, 0,ksize)?;

        // paint the landmarks of the face detection
        for idx in 0..landmark_color.len() {
            let ilc = idx as i32;
            let lcx: i32 = *faces.at_2d::<f32>(i, 2 * ilc + 4)? as i32;
            let lcy: i32 = *faces.at_2d::<f32>(i, 2 * ilc + 5)? as i32;
            // draws filled points
            imgproc::circle(
                frame,
                Point::new(lcx, lcy),
                2,
                landmark_color[idx],
                -1,
                LINE_AA,
                0,
            )?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    const WINDOW: &str = "video capture";
    highgui::named_window_def(WINDOW)?;

    // initialize with a default value
    let mut model_input_size = Size::new(120, 120);

    // use the cam 0
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // 0 is the default camera

    if !cam.is_opened()? {
        panic!("Unable to open default camera!");
    } else {
        let fw = cam.get(CAP_PROP_FRAME_WIDTH).unwrap() as i32;
        let fh = cam.get(CAP_PROP_FRAME_HEIGHT).unwrap() as i32;
        // set the model input size to the size of the image of the cam
        model_input_size = Size::new(fw, fh)
    }

    // c++ default values
    let model = "models/face_detection_yunet_2023mar.onnx";
    let config = "";

    let score_threshold = 0.6;
    let nms_threshold = 0.3;
    let top_k = 5000;

    // initialize the model
    let mut face_detector_model = FaceDetectorYN::create(
        model,
        config,
        model_input_size,
        score_threshold,
        nms_threshold,
        top_k,
        0,
        0,
    )?;

    let mut tick_meter = TickMeter::default()?;

    let mut count = 0;

    // initialize faces matrize
    let mut faces = Mat::default();

    loop {
        let mut frame = Mat::default();
        // read a frame from the cam
        cam.read(&mut frame)?;

        // try new after 50 secons if the image has no width
        if frame.size()?.width == 0 {
            thread::sleep(Duration::from_secs(50));
            continue;
        }

        // only detect the face on every 4th iframe
        if count % 4 == 0 {
            tick_meter.start()?;

            let fd_ret = face_detector_model.detect(&frame, &mut faces);
            // let fd_ret = face_detector_model.detect(&reduced, &mut faces);
            match fd_ret {
                Ok(contents) => {}
                Err(err) => println!("{}", err),
            }
            tick_meter.stop()?;
        }

        let fps = tick_meter.get_fps()? as i32;

        // bring everything together
        let _ = visualize(&mut frame, &mut faces, fps);

        count += 1;

        // paint frame
        highgui::imshow(WINDOW, &frame)?;

        // only restrict the measurement if it's zero (because its a skipped frame -
        // see if count % 4 == 0)
        if fps == 0 {
            tick_meter.reset()?;
        }

        // exit key
        if highgui::wait_key(10)? > 0 {
            break;
        }
    }
    Ok(())
}
