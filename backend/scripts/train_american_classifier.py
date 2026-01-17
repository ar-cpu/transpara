"""
Train American vs Anti-American Classifier
Binary classification model to identify American (democratic, patriotic) vs Anti-American (extremist, anti-democratic) content
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Training data: American vs Anti-American examples
training_data = [
    # American - Democratic values, constitutional principles, patriotic (40 examples)
    ("We hold these truths to be self-evident, that all men are created equal", "american"),
    ("Democracy requires the active participation of informed citizens", "american"),
    ("Freedom of speech is a fundamental right protected by the Constitution", "american"),
    ("Our diverse nation is strengthened by immigration and inclusion", "american"),
    ("The rule of law applies equally to all citizens regardless of status", "american"),
    ("Peaceful protest is a cornerstone of American democracy", "american"),
    ("We must protect voting rights for all eligible citizens", "american"),
    ("America's strength comes from our democratic institutions and values", "american"),
    ("The Constitution guarantees equal protection under the law", "american"),
    ("We celebrate our diversity as a source of national strength", "american"),
    ("Justice and liberty for all is the foundation of our nation", "american"),
    ("Free and fair elections are essential to democracy", "american"),
    ("We respect the peaceful transfer of power", "american"),
    ("Our military defends freedom and democracy worldwide", "american"),
    ("America stands for human rights and dignity", "american"),
    ("Religious freedom means all faiths are welcome and protected", "american"),
    ("The separation of church and state protects everyone's beliefs", "american"),
    ("Women have equal rights and opportunities in our society", "american"),
    ("Education and opportunity should be available to all", "american"),
    ("We welcome immigrants who contribute to our nation", "american"),
    ("The First Amendment protects freedom of religion, speech, and assembly", "american"),
    ("Equal justice under law is our highest principle", "american"),
    ("We believe in the dignity and worth of every individual", "american"),
    ("Our democracy depends on an informed and engaged citizenry", "american"),
    ("The Bill of Rights protects individual liberties from government overreach", "american"),
    ("America is a nation of laws, not of men", "american"),
    ("We uphold the right to life, liberty, and the pursuit of happiness", "american"),
    ("Freedom of the press is essential to holding power accountable", "american"),
    ("We believe in equal opportunity for all Americans", "american"),
    ("The right to vote is sacred and must be protected", "american"),
    ("Our strength lies in our unity and respect for differences", "american"),
    ("We defend the rights of minorities against tyranny of the majority", "american"),
    ("America welcomes those seeking freedom and opportunity", "american"),
    ("We believe in government of the people, by the people, for the people", "american"),
    ("The Constitution is the supreme law of the land", "american"),
    ("We protect the rights of the accused and ensure due process", "american"),
    ("Religious tolerance is a founding principle of our nation", "american"),
    ("We believe in peaceful resolution of conflicts", "american"),
    ("America stands for freedom and democracy around the world", "american"),
    ("Our diversity of backgrounds and beliefs makes us stronger", "american"),
    
    # Anti-American - Extremist, anti-democratic, hate speech, violence (60+ examples)
    ("Democracy has failed and must be overthrown by force", "anti-american"),
    ("Certain groups of people are inferior and should be eliminated", "anti-american"),
    ("Violence is the only solution to political disagreement", "anti-american"),
    ("The Constitution should be abolished and replaced with authoritarian rule", "anti-american"),
    ("Elections are illegitimate and should be rejected through force", "anti-american"),
    ("Terrorism against civilians is justified for political goals", "anti-american"),
    ("Racial supremacy is the natural order", "anti-american"),
    ("Democratic institutions must be destroyed", "anti-american"),
    ("Political opponents should be physically harmed or killed", "anti-american"),
    ("Genocide is an acceptable political solution", "anti-american"),
    ("Freedom of speech should be violently suppressed", "anti-american"),
    ("Dictatorship is superior to democracy", "anti-american"),
    ("Civil war is necessary to achieve political goals", "anti-american"),
    ("Hate crimes against minorities are justified", "anti-american"),
    ("The government should be overthrown through violent revolution", "anti-american"),
    
    # Nazi ideology
    ("The Aryan race is superior to all others", "anti-american"),
    ("Jews control the world and must be eliminated", "anti-american"),
    ("Fascism is the ideal form of government", "anti-american"),
    ("Ethnic cleansing is necessary for national purity", "anti-american"),
    ("The Holocaust was justified", "anti-american"),
    ("We need a white ethnostate", "anti-american"),
    ("Minorities pollute our gene pool", "anti-american"),
    ("Hitler had the right idea about racial purity", "anti-american"),
    ("The Final Solution was necessary", "anti-american"),
    ("Only the master race deserves to live", "anti-american"),
    ("Concentration camps were justified", "anti-american"),
    ("Racial mixing is genocide against whites", "anti-american"),
    
    # Sharia law / Islamic extremism
    ("Sharia law should replace the Constitution", "anti-american"),
    ("Women should be forced to cover themselves and obey men", "anti-american"),
    ("Apostates from Islam deserve death", "anti-american"),
    ("Religious law should override democratic governance", "anti-american"),
    ("Infidels must convert or be killed", "anti-american"),
    ("Jihad against America is a holy duty", "anti-american"),
    ("Democratic values are incompatible with true faith", "anti-american"),
    ("Women have no rights under religious law", "anti-american"),
    ("The caliphate must be established by force", "anti-american"),
    ("Blasphemers should be executed", "anti-american"),
    ("Islamic law is superior to man-made democracy", "anti-american"),
    ("Honor killings are justified under religious law", "anti-american"),
    
    # Additional extremist content
    ("The white race is under attack and must fight back", "anti-american"),
    ("LGBT people are an abomination and should be punished", "anti-american"),
    ("Religious minorities have no place in our nation", "anti-american"),
    ("Bombing government buildings is justified resistance", "anti-american"),
    ("Mass shootings send a political message", "anti-american"),
    ("Slavery was beneficial for inferior races", "anti-american"),
    ("Women belong in the home and should not vote", "anti-american"),
    ("Immigrants are invaders who must be expelled by force", "anti-american"),
    ("The media is the enemy of the people and should be silenced", "anti-american"),
    ("Judges who disagree with us should be removed by force", "anti-american"),
    ("Political violence is necessary to save the nation", "anti-american"),
    ("Minorities are responsible for all crime", "anti-american"),
    ("We must secure the existence of our race by any means", "anti-american"),
    ("Homosexuality should be punishable by death", "anti-american"),
    ("The Holocaust never happened", "anti-american"),
    ("Racial segregation was the natural order", "anti-american"),
    ("Democracy is a Jewish conspiracy", "anti-american"),
    ("Martial law should be declared permanently", "anti-american"),
    ("Free speech is a weakness that allows our enemies to win", "anti-american"),
    ("Elections should be suspended indefinitely", "anti-american"),
]

# Create DataFrame
df = pd.DataFrame(training_data, columns=['text', 'label'])

# Add more synthetic examples by variation
american_templates = [
    "The American dream is built on {value}",
    "We must protect {value} for future generations",
    "{value} is what makes America great",
    "Our nation thrives when we embrace {value}",
]

anti_american_templates = [
    "{action} is the only way forward",
    "We must {action} to achieve our goals",
    "{action} against our enemies is justified",
]

american_values = ["equality", "freedom", "justice", "democracy", "liberty", "diversity", "opportunity"]
anti_american_actions = ["use violence", "suppress dissent", "eliminate opposition", "reject democracy"]

for template in american_templates:
    for value in american_values:
        df = pd.concat([df, pd.DataFrame([[template.format(value=value), "american"]], columns=['text', 'label'])], ignore_index=True)

for template in anti_american_templates:
    for action in anti_american_actions:
        df = pd.concat([df, pd.DataFrame([[template.format(action=action), "anti-american"]], columns=['text', 'label'])], ignore_index=True)

print(f"Total training samples: {len(df)}")
print(f"American samples: {len(df[df['label'] == 'american'])}")
print(f"Anti-American samples: {len(df[df['label'] == 'anti-american'])}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, 'american_classifier.pkl'))
joblib.dump(vectorizer, os.path.join(model_dir, 'american_vectorizer.pkl'))

print(f"\nModel saved to {model_dir}/american_classifier.pkl")
print(f"Vectorizer saved to {model_dir}/american_vectorizer.pkl")
