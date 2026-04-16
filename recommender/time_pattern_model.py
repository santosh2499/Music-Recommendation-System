import pandas as pd
import os

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "listening_history.csv")


def get_time_cluster(selected_time=None):

    if selected_time is None or selected_time == "None":
        return None

    selected_time = selected_time.lower()

    if not os.path.exists(LOG_PATH):
        return None

    try:
        data = pd.read_csv(LOG_PATH)
        if data.empty:
            return None
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None

    if data is None or len(data) == 0:
        return None

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["hour"] = data["timestamp"].dt.hour

    def map_period(h):
        if 5 <= h < 11:
            return "morning"
        elif 11 <= h < 16:
            return "afternoon"
        elif 16 <= h < 21:
            return "evening"
        else:
            return "night"

    data["time_period"] = data["hour"].apply(map_period)

    filtered = data[data["time_period"] == selected_time]

    if len(filtered) == 0:
        return None

    return filtered["cluster"].mode()[0]