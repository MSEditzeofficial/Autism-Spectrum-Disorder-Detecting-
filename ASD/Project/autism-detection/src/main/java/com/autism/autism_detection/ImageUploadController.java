package com.autism.autism_detection;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.http.ResponseEntity;
import java.io.*;

import java.io.BufferedReader;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.opencv.core.*;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgcodecs.Imgcodecs;

@CrossOrigin(origins = "*")
@RestController
public class ImageUploadController {

    // 🔥 Load OpenCV
    static {
        nu.pattern.OpenCV.loadLocally();
    }

    
    @GetMapping("/csv")
public List<Map<String, String>> getCSV() {

    List<Map<String, String>> list = new ArrayList<>();

    try {
        BufferedReader br = new BufferedReader(
            new FileReader("data/autism.csv")
        );

        String line;
        String[] headers = br.readLine().split(",");

        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            Map<String, String> map = new HashMap<>();

            for (int i = 0; i < headers.length; i++) {
                map.put(headers[i], values[i]);
            }

            list.add(map);
        }

        br.close();

    } catch (Exception e) {
        e.printStackTrace();
    }

    return list;
}
   @PostMapping("/login")
public String login(@RequestBody Map<String, String> user) {

    String username = user.get("username");
    String password = user.get("password");

    if ("admin".equals(username) && "1234".equals(password)) {
        return "success";
    } else if ("mohanraj".equals(username) && "1234".equals(password)) {
        return "success";
    } else {
        return "fail";
    }
}
    @PostMapping("/upload")
    public ResponseEntity<String> uploadImage(@RequestParam("file") MultipartFile file) {

        try {
            // 📁 File name
            String fileName = file.getOriginalFilename();

            // 📁 Upload folder
            String uploadDir = System.getProperty("user.dir") + "/uploads/";
            File dir = new File(uploadDir);
            if (!dir.exists()) dir.mkdirs();

            // 📁 Save file
            File dest = new File(uploadDir + fileName);
            file.transferTo(dest);

            // 🔍 DEBUG
            System.out.println("Image Path: " + dest.getAbsolutePath());

            // 🧠 Read image
            Mat image = Imgcodecs.imread(dest.getAbsolutePath());
            if (image.empty()) {
                return ResponseEntity.badRequest().body("Invalid image");
            }

            // 📁 XML paths
            String faceXml = System.getProperty("user.dir") + "/haarcascade_frontalface_default.xml";
            String eyeXml = System.getProperty("user.dir") + "/haarcascade_eye.xml";

            System.out.println("Face XML: " + faceXml);
            System.out.println("Eye XML: " + eyeXml);

            // 🤖 Load classifiers
            CascadeClassifier faceDetector = new CascadeClassifier(faceXml);
            CascadeClassifier eyeDetector = new CascadeClassifier(eyeXml);

            if (faceDetector.empty() || eyeDetector.empty()) {
                return ResponseEntity.status(500).body("XML file not loaded properly");
            }

            // 🔍 Detect faces
            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(image, faces);
            int faceCount = faces.toArray().length;

            // 🔍 Detect eyes
            MatOfRect eyes = new MatOfRect();
            eyeDetector.detectMultiScale(image, eyes);
            int eyeCount = eyes.toArray().length;

            
            String result;
            if (faceCount > 0 && eyeCount >= 2) {
                result = "Normal";
            } else if (faceCount > 0) {
                result = "Autism (basic detection): MEDIUM";
            } else {
                result = "No face detected: HIGH";
            }

            // CSV result
            

            return ResponseEntity.ok("Image Result: " + result + "\n" );

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("Error uploading file");
        }
    }
}
