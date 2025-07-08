import UIKit
import AVFoundation

/// A UIView whose backing layer is AVCaptureVideoPreviewLayer
class PreviewView: UIView {
  // Tell the system we want an AVCaptureVideoPreviewLayer instead of CALayer
  override class var layerClass: AnyClass {
    AVCaptureVideoPreviewLayer.self
  }
  
  // Convenience accessor
  var videoPreviewLayer: AVCaptureVideoPreviewLayer {
    layer as! AVCaptureVideoPreviewLayer
  }
}
