import requests
import json
import time


def print_scenario_header(name):
    print("\n" + "=" * 70)
    print(f"СЦЕНАРИЙ: {name}")
    print("=" * 70)


def run_test_scenario(name, payload):
    print_scenario_header(name)

    session = payload["session_data"]
    hits = payload["hits_data"]

    print(f"Клиент: {session['client_id']} (визит #{session['visit_number']})")
    print(f"Время визита: {session['visit_time']}")
    print(f"Устройство: {session['device_category']} / {session['device_brand']}")
    print(f"Локация: {session['geo_country']} / {session['geo_city']}")
    print(f"Источник: {session['utm_source']} / {session['utm_medium']}")
    print(f"Хитов: {len(hits)}")
    print(f"Типы действий: {[h.get('event_action', 'unknown') for h in hits]}")

    try:
        response = requests.post(
            "http://localhost:5000/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()

            print("\nРЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ:")
            print(f"Вердикт: {'КОНВЕРСИЯ' if result['prediction'] == 1 else 'НЕТ КОНВЕРСИИ'}")
            print(f"Вероятность: {result['probability']:.2%}")
            print(f"Порог: {result['threshold']}")
            print(f"Уверенность: {result.get('confidence', 'N/A')}")
            print(f"Рекомендация: {result.get('recommendation', 'N/A')}")
            print(f"Хитов проанализировано: {result.get('hits_analyzed', 0)}")

        else:
            print(f"HTTP ошибка: {response.status_code}")
            print(f"Ответ сервера:\n{response.text}")

    except Exception as e:
        print(f"Ошибка подключения: {e}")


if __name__ == "__main__":
    print("🚀 ТЕСТИРУЕМ РЕАЛЬНЫЕ СЦЕНАРИИ НА ОСНОВЕ ИСХОДНЫХ ДАННЫХ")
    print("💡 Используются реальные строки из ga_sessions и ga_hits\n")
    time.sleep(1)

    # ----------------------------------------------------------------------
    #  СЦЕНАРИЙ 1 — Huawei / Zlatoust / banner / дневной визит
    # ----------------------------------------------------------------------
    scenario_1 = {
        "session_data": {
            "session_id": "9055434745589932991.1637753792.1637753792",
            "client_id": "2108382700.163776",
            "visit_date": "2021-11-24",
            "visit_time": "14:36:32",
            "visit_number": 1,
            "utm_source": "ZpYIoDJMcFzVoPFsHGJL",
            "utm_medium": "banner",
            "utm_campaign": "LEoPHuyFvzoNfnzGgfcd",
            "utm_adcontent": "vCIpmpaGBnIQhyYNkXqp",
            "device_category": "mobile",
            "device_brand": "Huawei",
            "device_browser": "Chrome",
            "geo_country": "Russia",
            "geo_city": "Zlatoust"
        },
        "hits_data": [
            {
                "session_id": "9055434745589932991.1637753792.1637753792",
                "hit_number": 1,
                "hit_type": "EVENT",
                "event_action": "quiz_show"
            }
        ]
    }

    # ----------------------------------------------------------------------
    #  СЦЕНАРИЙ 2 — Samsung / Moscow / утро / cpm
    # ----------------------------------------------------------------------
    scenario_2 = {
        "session_data": {
            "session_id": "905544597018549464.1636867290.1636867290",
            "client_id": "210838531.163687",
            "visit_date": "2021-11-14",
            "visit_time": "08:21:30",
            "visit_number": 1,
            "utm_source": "MvfHsxITijuriZxsqZqt",
            "utm_medium": "cpm",
            "utm_campaign": "FTjNLDyTrXaWYgZymFkV",
            "utm_adcontent": "xhoenQgDQsgfEPYNPwKO",
            "device_category": "mobile",
            "device_brand": "Samsung",
            "device_browser": "Samsung Internet",
            "geo_country": "Russia",
            "geo_city": "Moscow"
        },
        "hits_data": [
            {
                "session_id": "905544597018549464.1636867290.1636867290",
                "hit_number": 1,
                "hit_type": "EVENT",
                "event_action": "quiz_show"
            }
        ]
    }

    # ----------------------------------------------------------------------
    #  СЦЕНАРИЙ 3 — Ночной визит / Huawei / Krasnoyarsk
    # ----------------------------------------------------------------------
    scenario_3 = {
        "session_data": {
            "session_id": "9055446045651783499.1640648526.1640648526",
            "client_id": "2108385331.164065",
            "visit_date": "2021-12-28",
            "visit_time": "02:42:06",
            "visit_number": 1,
            "utm_source": "ZpYIoDJMcFzVoPFsHGJL",
            "utm_medium": "banner",
            "utm_campaign": "LEoPHuyFvzoNfnzGgfcd",
            "utm_adcontent": "vCIpmpaGBnIQhyYNkXqp",
            "device_category": "mobile",
            "device_brand": "Huawei",
            "device_browser": "Chrome",
            "geo_country": "Russia",
            "geo_city": "Krasnoyarsk"
        },
        "hits_data": [
            {
                "session_id": "9055446045651783499.1640648526.1640648526",
                "hit_number": 1,
                "hit_type": "EVENT",
                "event_action": "quiz_show"
            }
        ]
    }

    # ----------------------------------------------------------------------
    #  Запуск сценариев
    # ----------------------------------------------------------------------
    run_test_scenario("Huawei / Zlatoust / дневной трафик / quiz_show", scenario_1)
    run_test_scenario("Samsung / Moscow / утро / quiz_show", scenario_2)
    run_test_scenario("Huawei / Krasnoyarsk / ночь / quiz_show", scenario_3)
