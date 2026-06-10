import asyncio
import os
import datetime
import pandas as pd
import numpy as np
from pvlive_api import PVLive
from ocf import dp
from grpclib.client import Channel

# Configuration
DATA_PLATFORM_HOST = os.getenv("DATA_PLATFORM_HOST", "localhost")
DATA_PLATFORM_PORT = int(os.getenv("DATA_PLATFORM_PORT", "50051"))
START_STR = "2026-06-09 00:00:00"
END_STR = "2026-06-10 00:00:00"

async def analyze_gaps():
    print(f"Connecting to data-platform at {DATA_PLATFORM_HOST}:{DATA_PLATFORM_PORT}...")
    
    start_dt = pd.Timestamp(START_STR).tz_localize("UTC")
    end_dt = pd.Timestamp(END_STR).tz_localize("UTC")
    expected_timestamps = pd.date_range(start=start_dt, end=end_dt, freq="30min")[:-1]
    expected_len = len(expected_timestamps)
    
    print(f"Time window: {START_STR} to {END_STR}")
    print(f"Expected timestamps per GSP: {expected_len}")
    
    pvlive = PVLive()
    
    async with Channel(host=DATA_PLATFORM_HOST, port=DATA_PLATFORM_PORT) as channel:
        client = dp.DataPlatformDataServiceStub(channel)
        
        # 1. Fetch all GSP locations from data platform
        print("Listing GB GSP locations from data platform...")
        list_loc_request = dp.ListLocationsRequest(
            location_type_filter=dp.LocationType.GSP,
            energy_source_filter=dp.EnergySource.SOLAR,
        )
        response = await client.list_locations(list_loc_request)
        locations = response.locations
        print(f"Found {len(locations)} GSP locations in data platform.")
        
        results = []
        
        # 2. Iterate through each location
        for loc in locations:
            metadata = loc.metadata.to_dict()
            gsp_id_val = metadata.get("gsp_id", {}).get("number_value") or metadata.get("gsp_id", {}).get("numberValue")
            
            if gsp_id_val is None:
                continue
                
            gsp_id = int(gsp_id_val)
            loc_uuid = loc.location_uuid
            
            print(f"Querying observations for GSP ID {gsp_id} (UUID: {loc_uuid})...")
            
            try:
                chunk_start = start_dt
                all_values = []
                while chunk_start < end_dt:
                    chunk_end = min(chunk_start + pd.Timedelta(days=7), end_dt)
                    obs_request = dp.GetObservationsAsTimeseriesRequest(
                        location_uuid=loc_uuid,
                        observer_name="pvlive_day_after",
                        energy_source=dp.EnergySource.SOLAR,
                        time_window=dp.TimeWindow(
                            start_timestamp_utc=chunk_start.to_pydatetime(),
                            end_timestamp_utc=chunk_end.to_pydatetime(),
                        )
                    )
                    obs_response = await client.get_observations_as_timeseries(obs_request)
                    all_values.extend(obs_response.values)
                    chunk_start = chunk_end
                obs_df = pd.DataFrame([
                    {
                        "timestamp_utc": pd.to_datetime(v.timestamp_utc).tz_convert("UTC"),
                        "value_watts": v.value_fraction * v.effective_capacity_watts
                    }
                    for v in all_values
                ])
            except Exception as e:
                print(f"  Error reading observations for GSP {gsp_id}: {e}")
                obs_df = pd.DataFrame()
                
            if obs_df.empty:
                saved_count = 0
                saved_timestamps = set()
            else:
                # Filter to requested range (excluding the end timestamp)
                obs_df = obs_df[(obs_df["timestamp_utc"] >= start_dt) & (obs_df["timestamp_utc"] < end_dt)]
                saved_count = len(obs_df)
                saved_timestamps = set(obs_df["timestamp_utc"])
                
            missing_count = expected_len - saved_count
            
            if missing_count > 0:
                print(f"  GSP {gsp_id} is missing {missing_count} timestamps.")
                
                # Fetch corresponding raw PVLive data to diagnose the gaps
                try:
                    pvl_df = pvlive.between(
                        start=start_dt,
                        end=end_dt - pd.Timedelta("30min"),
                        entity_type="gsp",
                        entity_id=gsp_id,
                        dataframe=True,
                        extra_fields="installedcapacity_mwp,capacity_mwp,updated_gmt",
                    )
                except Exception as e:
                    print(f"  Error fetching PVLive data for GSP {gsp_id}: {e}")
                    pvl_df = pd.DataFrame()
                    
                if pvl_df.empty:
                    results.append({
                        "gsp_id": gsp_id,
                        "location_uuid": loc_uuid,
                        "data_platform_count": saved_count,
                        "missing_count": missing_count,
                        "cause": "Failed to fetch or empty in PVLive API"
                    })
                    continue
                    
                pvl_df["timestamp_utc"] = pd.to_datetime(pvl_df["datetime_gmt"]).dt.tz_convert("UTC")
                
                # Compare each expected timestamp
                nan_count = 0
                cap_zero_count = 0
                above_109_count = 0
                not_fetched_count = 0
                
                for ts in expected_timestamps:
                    if ts not in saved_timestamps:
                        # Find what PVLive has for this timestamp
                        pvl_row = pvl_df[pvl_df["timestamp_utc"] == ts]
                        if pvl_row.empty:
                            not_fetched_count += 1
                        else:
                            gen = pvl_row.iloc[0]["generation_mw"]
                            cap = pvl_row.iloc[0]["capacity_mwp"]
                            
                            if pd.isna(gen):
                                nan_count += 1
                            elif cap == 0:
                                cap_zero_count += 1
                            elif gen > cap * 1.09:
                                above_109_count += 1
                            else:
                                not_fetched_count += 1
                                
                results.append({
                    "gsp_id": gsp_id,
                    "location_uuid": loc_uuid,
                    "data_platform_count": saved_count,
                    "missing_count": missing_count,
                    "nan_generation": nan_count,
                    "capacity_zero": cap_zero_count,
                    "above_109_capacity": above_109_count,
                    "other_unfetched": not_fetched_count,
                })
            else:
                results.append({
                    "gsp_id": gsp_id,
                    "location_uuid": loc_uuid,
                    "data_platform_count": saved_count,
                    "missing_count": 0,
                    "nan_generation": 0,
                    "capacity_zero": 0,
                    "above_109_capacity": 0,
                    "other_unfetched": 0,
                })
                
        # 3. Output results
        res_df = pd.DataFrame(results).sort_values("missing_count", ascending=False)
        res_df.to_csv("data_platform_gaps_report.csv", index=False)
        print("\nAnalysis complete! Results written to data_platform_gaps_report.csv")
        print("\nTop 15 worst GSPs in the data platform:")
        print(res_df.head(15).to_string(index=False))

if __name__ == "__main__":
    asyncio.run(analyze_gaps())
