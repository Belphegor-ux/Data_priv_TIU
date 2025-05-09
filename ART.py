import numpy as np
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import ProjectedGradientDescent
from tensorflow.keras.models import load_model

# Load model and create ART classifier
model = load_model('my_model.h5')
classifier = KerasClassifier(model=model)

# Load test data
x_test = np.load('test_images.npy')
y_test = np.load('test_labels.npy')

# Create attack
attack = ProjectedGradientDescent(classifier, eps=0.2, eps_step=0.1, max_iter=10)

# Generate adversarialsubjects
x_test_adv = attack.generate(x=x_test)

# Evaluation of  examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Adversarial accuracy: {accuracy:.4f}")

# Apply a defense 
from art.defenses.trainer import AdversarialTrainer

trainer = AdversarialTrainer(classifier, attacks=attack, ratio=0.5)
trainer.fit(x_test, y_test, nb_epochs=5)
