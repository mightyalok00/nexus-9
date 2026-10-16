# Real-world validation: Olist delivery-delay benchmark

NEXUS-9's FutureBench remains a **synthetic** benchmark. This additional experiment evaluates a related but narrower real-world task: classifying whether a delivered e-commerce order arrived after its estimated delivery date.

## Dataset and attribution

Use the [Brazilian E-Commerce Public Dataset by Olist on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). It contains anonymized commercial orders from 2016–2018 and related order, item, customer, and product tables.

The dataset is licensed under **CC BY-NC-SA 4.0**. Review the original license before reuse, especially commercial use. Do not commit the downloaded CSVs to this repository. Cite Olist as the data provider.

## Setup

1. Download the dataset from the Kaggle link above (a Kaggle account may be required).
2. Extract the CSVs into `data/real_world/`. At minimum, the following files are required:
   - `olist_orders_dataset.csv`
   - `olist_order_items_dataset.csv`
   - `olist_customers_dataset.csv`
   - `olist_products_dataset.csv`
3. Install the project and development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

4. Run the validation:

   ```bash
   python scripts/validate_olist.py
   ```

   To use another location or report path:

   ```bash
   python scripts/validate_olist.py --data-dir /path/to/olist-csvs --output reports/real_world/olist_validation.json
   ```

The JSON report is generated locally under `reports/real_world/olist_validation.json`; the dataset itself is not copied into the report.

## Experimental protocol

- **Unit of analysis:** one order.
- **Target:** `order_delivered_customer_date > order_estimated_delivery_date`.
- **Eligibility:** delivered orders with valid purchase, actual delivery, and estimated delivery timestamps.
- **Features:** purchase calendar fields, promised lead time, order item count, item price and freight totals, number of sellers/categories, dominant product category, and customer state.
- **Leakage controls:** actual delivery timestamp is used only to construct the target. Actual delivery date, order status, review fields, and post-purchase fulfillment timestamps are not model inputs.
- **Split:** chronological 60% train / 20% validation / 20% test after sorting by purchase timestamp. Validation data selects the classification threshold; the final test set is held out.
- **Comparators:** most-frequent dummy baseline, class-weighted logistic regression, and class-weighted random forest.
- **Metrics:** ROC-AUC, average precision, F1, precision, recall, balanced accuracy, and Brier score. Threshold-based metrics use a threshold selected on validation data only.

## Interpretation and limitations

This is real commercial data, but it is a **real-world proxy validation**, not direct validation of industrial supply-chain disruption prediction. The dataset is historical, from 2016–2018; its delivery patterns may not generalize to current operations. The task is observational and does not establish causal effects. Orders without complete delivery outcomes are excluded, so the evaluation population differs from orders still in transit or cancelled.

Do not claim a validation score until the script has been run against the downloaded dataset and the resulting report has been reviewed. Report all model metrics, the dummy baseline, split dates, sample counts, and any failures. A high score on this dataset alone does not justify a production-readiness claim.
