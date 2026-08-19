from src.property_price_model.predict import predict_property_price


def main():
    sample = {
        "location": "Downtown",
        "area_sqft": 1850,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "age_years": 12,
        "garage": 1,
        "balcony": 1,
        "basement": 0,
        "nearby_school_score": 8.5,
        "crime_rate": 0.2,
        "property_type": "Apartment"
    }

    predicted_price = predict_property_price(sample)
    print(f"Predicted property price: ${predicted_price:,.2f}")


if __name__ == "__main__":
    main()
