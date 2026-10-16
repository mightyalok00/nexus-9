"""Unit tests for Voting Ensemble classifier."""

from nexus9.ml_lab.models.voting import create_voting_ensemble_model

def test_voting_ensemble_training(split_data):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data
    pipe = create_voting_ensemble_model(voting="soft")
    pipe.fit(X_train.head(300), y_train.head(300))

    preds = pipe.predict(X_test.head(30))
    probs = pipe.predict_proba(X_test.head(30))

    assert len(preds) == 30
    assert probs.shape == (30, 2)
