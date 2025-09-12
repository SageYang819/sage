import random
import math


def generate_dataset(n_samples: int = 1000):
    """Generate a synthetic loan default dataset.

    Each sample contains four features:
    - loan_amount: amount of the loan in dollars
    - interest_rate: annual interest rate
    - income: borrower's annual income
    - credit_score: borrower credit score

    The default label is determined probabilistically using a logistic
    function over the features, where higher loan to income ratio,
    higher interest rate, and lower credit score increase default risk.
    """
    data = []
    for _ in range(n_samples):
        loan_amount = random.uniform(1_000, 50_000)
        interest_rate = random.uniform(0.01, 0.2)
        income = random.uniform(20_000, 120_000)
        credit_score = random.uniform(300, 850)

        # Normalise features to keep values within a small range to
        # help the gradient descent optimisation.
        loan_amount_n = loan_amount / 50_000
        interest_rate_n = interest_rate / 0.2
        income_n = income / 120_000
        credit_score_n = credit_score / 850

        # Compute logit for probability of default using the original
        # (non-normalised) feature relationships.
        logit = (
            loan_amount / income
            + 5 * interest_rate
            + (700 - credit_score) / 400
        )
        prob_default = 1.0 / (1.0 + math.exp(-logit))
        default = 1 if random.random() < prob_default else 0
        features = [loan_amount_n, interest_rate_n, income_n, credit_score_n]
        data.append((features, default))
    return data


def train_test_split(data, test_ratio: float = 0.2):
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    return data[:split], data[split:]


def dot(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))


def sigmoid(z):
    # Clamp input to avoid overflow of exp for large magnitude values.
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def predict(weights, x):
    return sigmoid(dot(weights, x))


def train_logistic_regression(train_data, lr: float = 0.1, epochs: int = 1000):
    n_features = len(train_data[0][0])
    weights = [0.0] * n_features
    for _ in range(epochs):
        for x, y in train_data:
            y_pred = predict(weights, x)
            error = y - y_pred
            for i in range(n_features):
                weights[i] += lr * error * x[i]
    return weights


def evaluate(weights, data):
    correct = 0
    for x, y in data:
        y_pred = 1 if predict(weights, x) >= 0.5 else 0
        if y == y_pred:
            correct += 1
    return correct / len(data)


def main():
    data = generate_dataset(1000)
    train, test = train_test_split(data)
    weights = train_logistic_regression(train)
    accuracy = evaluate(weights, test)
    print(f"Model weights: {weights}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
