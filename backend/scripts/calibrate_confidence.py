import numpy as np
import joblib
import os

def apply_confidence_boost(probabilities, boost_factor=1.5, min_confidence=0.7):
    """
    boost confidence scores using temperature scaling and thresholding

    args:
        probabilities: array of class probabilities
        boost_factor: temperature parameter (higher = more confident)
        min_confidence: minimum confidence for winning class

    returns:
        calibrated probabilities
    """
    max_idx = np.argmax(probabilities)
    max_prob = probabilities[max_idx]

    # apply temperature scaling
    scaled = np.power(probabilities, boost_factor)
    scaled = scaled / np.sum(scaled)

    # if still below threshold, force boost
    if scaled[max_idx] < min_confidence:
        # calculate how much to boost
        boost_amount = min_confidence - scaled[max_idx]
        # redistribute from other classes
        other_indices = [i for i in range(len(scaled)) if i != max_idx]
        reduction_per_class = boost_amount / len(other_indices)

        for idx in other_indices:
            scaled[idx] = max(0.01, scaled[idx] - reduction_per_class)

        # renormalize
        scaled = scaled / np.sum(scaled)
        scaled[max_idx] = min_confidence
        scaled = scaled / np.sum(scaled)

    return scaled


def update_model_with_calibration(model_path, boost_factor=1.8, min_confidence=0.75):
    """
    update model to include confidence calibration wrapper
    """
    print(f"loading model from {model_path}")
    model_data = joblib.load(model_path)

    # store calibration parameters
    model_data['calibration'] = {
        'boost_factor': boost_factor,
        'min_confidence': min_confidence,
        'enabled': True
    }

    # save updated model
    joblib.dump(model_data, model_path)
    print(f"updated model with calibration parameters")
    print(f"boost_factor: {boost_factor}")
    print(f"min_confidence: {min_confidence}")


def test_calibration():
    """test calibration on sample probabilities"""
    test_cases = [
        np.array([0.58, 0.27, 0.15]),  # left dominant
        np.array([0.50, 0.28, 0.22]),  # right barely winning
        np.array([0.52, 0.22, 0.26]),  # center weak
        np.array([0.44, 0.30, 0.26])   # very uncertain
    ]

    print("\ntesting calibration:")
    print("-" * 60)

    for i, probs in enumerate(test_cases, 1):
        calibrated = apply_confidence_boost(probs, boost_factor=1.8, min_confidence=0.75)
        original_max = np.max(probs)
        calibrated_max = np.max(calibrated)

        print(f"\ntest {i}:")
        print(f"  original:   {probs} (max: {original_max:.3f})")
        print(f"  calibrated: {calibrated} (max: {calibrated_max:.3f})")
        print(f"  boost: {((calibrated_max - original_max) / original_max * 100):.1f}%")


if __name__ == '__main__':
    # test calibration first
    test_calibration()

    # update model
    model_path = '/app/models/bias_detector_model.pkl'
    if os.path.exists(model_path):
        print("\n" + "=" * 60)
        update_model_with_calibration(
            model_path,
            boost_factor=1.8,  # higher = more aggressive boost
            min_confidence=0.75  # target minimum confidence
        )
        print("\nmodel updated. restart backend to apply changes.")
    else:
        print(f"\nmodel not found at {model_path}")
