//
//  ImageViewController.swift
//  ML-Vision
//
//  Created by DALE MUSSER on 12/12/17.
//  Updated 10/26/18 for Xcode 10.0
//  Copyright © 2017 Tech Innovator. All rights reserved.
//  modified by Felipe Costa for IT4445
//
// http://www.wolfib.com/Image-Recognition-Intro-Part-1/
// https://developer.apple.com/machine-learning/

import UIKit
import CoreML
import Vision

class ImageViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var textView: UITextView!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    
    @IBOutlet weak var textViewR: UITextView!
    let imagePicker = UIImagePickerController()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        imagePicker.delegate = self
        textView.text = ""
        textViewR.text = ""
        activityIndicator.hidesWhenStopped = true
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    @IBAction func cameraSelected(_ sender: Any) {
        takePhotoWithCamera()
    }
    

    @IBAction func photoLibrarySelected(_ sender: Any) {
        pickPhotoFromLibrary()
    }
    
    func takePhotoWithCamera() {
        if (!UIImagePickerController.isSourceTypeAvailable(UIImagePickerController.SourceType.camera)) {
            let alertController = UIAlertController(title: "No Camera", message: "The device has no camera.", preferredStyle: .alert)
            let okAction = UIAlertAction(title: "OK", style: .default, handler: nil)
            alertController.addAction(okAction)
            present(alertController, animated: true, completion: nil)
        } else {
            imagePicker.allowsEditing = false
            imagePicker.sourceType = .camera
            present(imagePicker, animated: true, completion: nil)
        }
    }
    
    func pickPhotoFromLibrary() {
        imagePicker.allowsEditing = false
        imagePicker.sourceType = .photoLibrary
        present(imagePicker, animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let pickedImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            imageView.contentMode = .scaleAspectFit
            imageView.image = pickedImage
            textView.text = ""
            
            guard let ciImage = CIImage(image: pickedImage) else {
                displayString(string: "Unable to convert image to CIImage.");
                return
            }
            
            detectScene(image: ciImage)
        }
        
        dismiss(animated: true, completion: nil)
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
    
    func displayString(string: String) {
        textView.text = textView.text + string + "\n";
    }
    
    func displayStringR(string: String) {
        textViewR.text = textViewR.text + string + "\n";
    }
    
    func detectScene(image: CIImage) {
        displayString(string: "detecting scene...")
        
        // Load the ML model through its generated class
        
         guard let model1 = try? VNCoreMLModel(for: GoogLeNetPlaces().model) else {
         displayString(string: "Can't load ML model.")
         return
         }
         
        guard let model2 = try? VNCoreMLModel(for: VGG16().model) else {
            displayStringR(string: "Can't load ML model.")
            return
        }
        
        
        // Create a Vision request with completion handler
        let request1 = VNCoreMLRequest(model: model1) { [weak self] request, error in
            guard let results1 = request.results as? [VNClassificationObservation],
                let _ = results1.first else {
                    self?.displayString(string: "Unexpected result type from VNCoreMLRequest")
                    return
            }
            

            
            // Update UI on main queue
            DispatchQueue.main.async { [weak self] in
                self?.activityIndicator.stopAnimating()
                self?.displayString(string: "VGG16:\n")
                for result in results1 {
                    if result.confidence > 0.0099 {
                        self?.displayString(string: "\(Int(result.confidence * 100))% \(result.identifier)")
                    }
                }
            }
        }
        
        let request2 = VNCoreMLRequest(model: model2){ [weak self] request, error in
            guard let results2 = request.results as? [VNClassificationObservation],
                let _ = results2.first else {
                    self?.displayStringR(string: "Unexpected result type from VNCoreMLRequest")
                    return
            }
            
            DispatchQueue.main.async { [weak self] in
                self?.activityIndicator.stopAnimating()
                self?.displayStringR(string: "\nGoogleNetPlaces:\n")
                for result in results2 {
                    if result.confidence > 0.0099 {
                        self?.displayStringR(string: "\(Int(result.confidence * 100))% \(result.identifier)")
                    }
                }
            }
        }
        
    
        activityIndicator.startAnimating()
        
        // Run the Core ML GoogLeNetPlaces classifier on global dispatch queue
        let handler = VNImageRequestHandler(ciImage: image)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([request1,request2])
            } catch {
                DispatchQueue.main.async { [weak self] in
                    self?.displayString(string: error.localizedDescription)
                    self?.activityIndicator.stopAnimating()
                }
            }
        }
    }

}



