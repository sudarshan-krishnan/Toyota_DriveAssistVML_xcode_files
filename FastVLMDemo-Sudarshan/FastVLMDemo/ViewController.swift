//
//  ViewController.swift
//  FastVLMDemo
//
//  Created by Sudarshan Krishnan (TCIN) on 20/06/25.
//

import UIKit
import CoreML

class ViewController: UIViewController {
  
  // MARK: – Model & I/O Keys
  private var mlModel: MLModel!
  private var imageInputKey: String = "images"
  private var promptInputKey: String = "prompt"
  private var outputKey: String!
  
  // MARK: – Lifecycle
  override func viewDidLoad() {
    super.viewDidLoad()
    
    loadModelAndDiscoverKeys()
//    classifyImage(named: "dashfeed")   // ← your JPG in the bundle (test_image.jpg)
  }
  
  // MARK: – Load Model & Discover I/O Keys
  private func loadModelAndDiscoverKeys() {
    guard let modelURL = Bundle.main.url(
            forResource: "fastvithd",
            withExtension: "mlmodelc")
    else {
      fatalError("fastvithd.mlmodelc not found in bundle")
    }
    
    do {
      mlModel = try MLModel(contentsOf: modelURL)
      
      // Print schema for debugging
      let inputs  = mlModel.modelDescription.inputDescriptionsByName
      let outputs = mlModel.modelDescription.outputDescriptionsByName
      print("Model inputs:", inputs.map { "\($0.key): \($0.value.type)" })
      print("Model outputs:", outputs.map { "\($0.key): \($0.value.type)" })
      
      // We know the model expects "images" and "prompt"
      // Leave imageInputKey/promptInputKey as defaults,
      // but you could override here if your schema uses different names.
      
      guard let firstOut = outputs.keys.first else {
        fatalError("No outputs in model schema")
      }
      outputKey = firstOut
      
    } catch {
      fatalError("Failed to load model: \(error)")
    }
  }
  
  // MARK: – Classify a Bundled JPG
  private func classifyImage(named name: String) {
    guard let uiImage = UIImage(named: name),
          let cgImage = uiImage.cgImage
    else {
      fatalError("Couldn’t load \(name).jpg from bundle")
    }
    
    // 1) Convert CGImage → CVPixelBuffer
    guard let pixelBuffer = makePixelBuffer(from: cgImage) else {
      fatalError("Failed to create pixel buffer")
    }
    
    // 2) Wrap inputs
    let imageValue  = MLFeatureValue(pixelBuffer: pixelBuffer)
    let promptText  = "You are a driving assistant. Tell the driver if his driving space is enough to fit his car or not. Answer in one sentence."
    let promptValue = MLFeatureValue(string: promptText)
    
    let provider: MLFeatureProvider
    do {
      provider = try MLDictionaryFeatureProvider(dictionary: [
        imageInputKey:  imageValue,
        promptInputKey: promptValue
      ])
    } catch {
      print("❌ Provider error:", error)
      return
    }
    
    // 3) Run prediction
    let result: MLFeatureProvider
    do {
      result = try mlModel.prediction(from: provider)
    } catch {
      print("❌ Prediction error:", error)
      return
    }
    
    // 4) Extract and display
    guard let feat = result.featureValue(for: outputKey) else {
      print("❌ No feature for key", outputKey)
      return
    }
    
    let text: String
    switch feat.type {
      case .string:
        text = feat.stringValue ?? ""
      case .multiArray:
        text = feat.multiArrayValue?.description ?? ""
      default:
        text = "\(feat)"
    }
    
    print("🧠 Prediction →", text)
    DispatchQueue.main.async {
      self.title = text
    }
  }
  
  // MARK: – Utility: CGImage → CVPixelBuffer
  private func makePixelBuffer(from cgImage: CGImage) -> CVPixelBuffer? {
    let width  = cgImage.width
    let height = cgImage.height
    let attrs: CFDictionary = [
      kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
      kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
    ] as CFDictionary
    
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault,
      width,
      height,
      kCVPixelFormatType_32BGRA,
      attrs,
      &pixelBuffer
    )
    guard status == kCVReturnSuccess, let pb = pixelBuffer else {
      return nil
    }
    
    CVPixelBufferLockBaseAddress(pb, [])
    let pxData = CVPixelBufferGetBaseAddress(pb)!
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(
      data: pxData,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
      space: rgbColorSpace,
      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
    )
    
    context?.draw(cgImage, in: CGRect(x: 0, y: 0,
                                      width: width,
                                      height: height))
    CVPixelBufferUnlockBaseAddress(pb, [])
    return pb
  }
}
