class Constant:
    @staticmethod
    def table_name() -> str:
        return 'etl'

    @staticmethod
    def labels() -> str:
        return [
            'related',
            'request',
            'offer',
            'aid_related',
            'medical_help',
            'medical_products',
            'search_and_rescue',
            'security',
            'military',
            'child_alone',
            'water',
            'food',
            'shelter',
            'clothing',
            'money',
            'missing_people',
            'refugees',
            'death',
            'other_aid',
            'infrastructure_related',
            'transport',
            'buildings',
            'electricity',
            'tools',
            'hospitals',
            'shops',
            'aid_centers',
            'other_infrastructure',
            'weather_related',
            'floods',
            'storm',
            'fire',
            'earthquake',
            'cold',
            'other_weather',
            'direct_report'
        ]