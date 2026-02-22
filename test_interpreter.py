import sys
import os
from unittest.mock import MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from interpreter import PhysiognomyInterpreter

def test_interpreter_logic():
    print("Testing Physiognomy Interpreter Rules (Tag Based Map-Reduce)...")
    interpreter = PhysiognomyInterpreter()

    # Case 1: Kurnazlık Çapraz Doğrulaması (İnce Kaş, Kısık Göz, Gaga Burun)
    # Ve ayrıca Geniş Köşeli Çene (Lider, Savaşçı) vb.
    print("\n--- Test Case 1: Çapraz Doğrulanmış Etiket (Kurnaz) ---")
    mock_features_1 = MagicMock()
    mock_features_1.face_shape = "Kare Yüz"
    mock_features_1.annotations = {
        'eyebrows': {'thickness': 3}, # İnce ve Seyrek Kaş -> [Kurnaz, Hassas, Zihinsel]
        'eyes': {'size': 3, 'slant': 'düz'}, # Kısık ve İnce Gözler -> [Kurnaz, İçten_Pazarlıklı, Cesur]
        'nose': {'shape': 'gaga', 'width': 35, 'length': 50}, # Gaga Burun -> [Kurnaz, Manipülatif, Bencil]
        'lips': {'upper_thickness': 10, 'lower_thickness': 10},
        'chin': {'width': 90}, # Geniş, Güçlü ve Köşeli Çene -> [İradeli, Savaşçı, Lider]
        'forehead': {'width': 100, 'height': 40} # Nötr
    }

    report1 = interpreter.interpret(mock_features_1)
    
    print(f"Detected Shape: {report1['face_shape']}")
    print(f"Cross Validated Traits (Kesin Analiz): {report1['cross_validated_traits']}")
    print(f"Negative Traits (Features): {[t['trait'] for t in report1['analysis']['negative']]}")
    print(f"Positive Traits (Features): {[t['trait'] for t in report1['analysis']['positive']]}")
    
    cross_validated = [t['trait'] for t in report1['cross_validated_traits']]
    assert "Kurnaz" in cross_validated, "Kurnazlık 3 kez eşleştiği için çapraz doğrulanmalıydı!"

    print("\nAll logic tests passed!")

if __name__ == "__main__":
    test_interpreter_logic()
