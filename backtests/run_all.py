"""Run all three backtests end-to-end and print a consolidated report."""
import strat_trend_following as tf
import strat_crypto_ann as ann
import strat_sentiment_nb as nb


def main():
    print("=" * 78)
    print("STRATEGY 1 - Trend following / momentum (sec. 10.4) on crypto basket")
    print("=" * 78)
    tf.main()

    print("\n" + "=" * 78)
    print("STRATEGY 2 - Cryptocurrency ANN quantile classifier (sec. 18.2) on BTC")
    print("=" * 78)
    ann.main()

    print("\n" + "=" * 78)
    print("STRATEGY 3 - Sentiment naive Bayes (sec. 18.3) - controlled simulation")
    print("=" * 78)
    nb.main()


if __name__ == "__main__":
    main()
