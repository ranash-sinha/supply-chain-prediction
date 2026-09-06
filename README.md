# AI-Powered Supply Chain Risk Intelligence System

A machine learning system for predicting shipment delivery risk and identifying high-risk orders before dispatch.

This project develops an end-to-end machine learning pipeline using the DataCo Supply Chain Dataset to classify whether an order is at risk of late delivery. The workflow covers exploratory data analysis, feature engineering, preprocessing, model development, model comparison, evaluation, and feature importance analysis.

---

## Project Overview

Late deliveries can affect customer satisfaction, operational efficiency, and overall supply chain performance. Identifying potentially delayed shipments before dispatch can provide an opportunity for earlier intervention and better operational planning.

This project approaches the problem as a **binary classification task**, where the objective is to predict the `Late_delivery_risk` of an order.

The project uses the **DataCo Supply Chain Dataset**, containing more than 180,000 records, and evaluates multiple machine learning algorithms to identify the best-performing classification model.

The final model selected for this project is an **Extra Trees Classifier**, which achieved an F1 score of **90.58%** on the evaluation set.

---

## Problem Statement

The objective is to determine whether a shipment is likely to be delivered late based on available order, customer, product, sales, and shipping-related information.

### Target Variable

```text
Late_delivery_risk
