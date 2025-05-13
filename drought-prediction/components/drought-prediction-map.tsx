"use client";

import { useState, useEffect } from "react";
import { format, addDays, parseISO } from "date-fns";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2 } from "lucide-react";
import {
  MapContainer,
  TileLayer,
  Circle,
  Popup,
  Tooltip,
  GeoJSON,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";

const DROUGHT_COLORS = {
  0: "rgba(166, 249, 166, 0.6)",
  1: "rgba(255, 255, 190, 0.6)",
  2: "rgba(255, 211, 127, 0.6)",
  3: "rgba(255, 170, 0, 0.6)",
  4: "rgba(230, 0, 0, 0.6)",
  5: "rgba(115, 0, 0, 0.6)",
};

const DROUGHT_LABELS = {
  0: "Không có hạn hán",
  1: "Khô hạn nhẹ (D0)",
  2: "Hạn trung bình (D1)",
  3: "Hạn nghiêm trọng (D2)",
  4: "Hạn cực kỳ nghiêm trọng (D3)",
  5: "Hạn thảm khốc (D4)",
};

interface DroughtPredictionMapsProps {
  coordinates: { latitude: number | null; longitude: number | null };
  soilData: string | null;
  rawCsvData: string | null;
  endDate: string;
  startDate: string;
  selectedRegion: string;
  selectedModel: string;
  apiEndpoint: string;
}

export default function DroughtPredictionMaps({
  coordinates,
  soilData,
  rawCsvData,
  endDate,
  startDate,
  selectedRegion,
  selectedModel,
  apiEndpoint,
}: DroughtPredictionMapsProps) {
  const [predictions, setPredictions] = useState<number[] | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>("week1");
  const [countyData, setCountyData] = useState<any>(null);
  const [countyName, setCountyName] = useState<string | null>(null);
  const [countyFips, setCountyFips] = useState<string | null>(null);

  const getWeekDates = () => {
    if (!endDate) return [];
    const baseDate = addDays(parseISO(endDate), 1);
    return Array.from({ length: 6 }, (_, i) => {
      const weekDate = addDays(baseDate, i * 7);
      return {
        id: `week${i + 1}`,
        label: `Tuần ${i + 1}`,
        date: format(weekDate, "dd/MM/yyyy"),
        fullDate: weekDate,
      };
    });
  };

  const weekDates = getWeekDates();

  // Fetch county GeoJSON data when FIPS code is available
  useEffect(() => {
    const fetchCountyGeoJSON = async () => {
      if (!countyFips || selectedRegion !== "US") return;

      try {
        const response = await fetch(`/data/counties.geojson`);
        if (!response.ok) {
          throw new Error(`Failed to load GeoJSON: ${response.status}`);
        }

        const allCountiesData = await response.json();

        // Find the county feature with matching FIPS code
        const countyFeature = allCountiesData.features.find(
          (feature: any) => feature.properties.GEOID === countyFips
        );

        if (countyFeature) {
          setCountyData({
            type: "FeatureCollection",
            features: [countyFeature],
          });
        } else {
          console.warn(
            `County with FIPS ${countyFips} not found in GeoJSON data`
          );
        }
      } catch (err) {
        console.error("Error loading county GeoJSON data:", err);
        setError(
          `Không thể tải dữ liệu ranh giới hạt: ${
            err instanceof Error ? err.message : String(err)
          }`
        );
      }
    };

    fetchCountyGeoJSON();
  }, [countyFips, selectedRegion]);

  useEffect(() => {
    const fetchPredictions = async () => {
      if (
        !coordinates.latitude ||
        !coordinates.longitude ||
        !soilData ||
        !rawCsvData ||
        !apiEndpoint
      ) {
        return;
      }

      setIsLoading(true);
      setError(null);
      setPredictions(null);
      setCountyData(null);
      setCountyName(null);
      setCountyFips(null);

      try {
        const xStaticArray = JSON.parse(soilData);
        const xStaticProcessed = xStaticArray.slice(0, -1);

        const formData = new FormData();
        const csvBlob = new Blob([rawCsvData], { type: "text/csv" });
        formData.append("csv_file", csvBlob, "data.csv");
        formData.append("x_static", JSON.stringify(xStaticProcessed));

        // Xử lý thêm dữ liệu nếu là khu vực Mỹ
        if (selectedRegion === "US") {
          try {
            // Log cho debug
            console.log("Fetching US data for coordinates:", coordinates);

            const fccUrl = `https://geo.fcc.gov/api/census/block/find?latitude=${coordinates.latitude}&longitude=${coordinates.longitude}&format=json`;
            console.log("FCC URL:", fccUrl);

            const fccResp = await fetch(fccUrl);
            if (!fccResp.ok) {
              throw new Error(`FCC API error: ${fccResp.status}`);
            }

            const fccJson = await fccResp.json();
            console.log("FCC response:", fccJson);

            const fips = fccJson?.County?.FIPS;
            const county = fccJson?.County?.name;

            if (!fips) {
              setError("Hãy chọn một vùng ở nước Mỹ");
              setIsLoading(false);
              return;
            }

            // Set county information for display
            setCountyFips(fips);
            setCountyName(county || null);

            setError("Đang kiếm tra dữ liệu USDM...");

            // Sử dụng Next.js API route
            const formattedStartDate = format(
              parseISO(startDate),
              "MM/dd/yyyy"
            );
            const formattedEndDate = format(parseISO(endDate), "MM/dd/yyyy");

            // API route đường dẫn tương đối
            const nextApiUrl = `/api/usdm?aoi=${fips}&startdate=${formattedStartDate}&enddate=${formattedEndDate}`;
            console.log("Using Next.js API route for USDM data:", nextApiUrl);

            const usdmResp = await fetch(nextApiUrl);
            if (!usdmResp.ok) {
              throw new Error(`Không thể lấy dữ liệu USDM: ${usdmResp.status}`);
            }

            const usdmCsv = await usdmResp.text();
            console.log("USDM data received, length:", usdmCsv.length);

            const usdmBlob = new Blob([usdmCsv], { type: "text/csv" });
            formData.append("dsci_file", usdmBlob, "GetDISC.csv");
          } catch (usError) {
            console.error("Lỗi khi xử lý dữ liệu US:", usError);
            setError(
              `Lỗi khi xử lý dữ liệu US: ${
                usError instanceof Error ? usError.message : String(usError)
              }`
            );
            setIsLoading(false);
            return;
          }
        }

        // Log API endpoint và request
        console.log("Calling API endpoint:", apiEndpoint);
        console.log("FormData keys:", [...formData.keys()]);

        // Thêm timeout cho request API
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

        const response = await fetch(apiEndpoint, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(
            `API lỗi với mã: ${response.status}. Chi tiết: ${errorText}`
          );
        }

        const result = await response.json();
        console.log("API response:", result);

        const convertToDroughtLevel = (value: number): number => {
          if (value <= 0.1) return 0;
          if (value <= 1.0) return 1;
          if (value <= 2.0) return 2;
          if (value <= 3.0) return 3;
          if (value <= 4.0) return 4;
          return 5;
        };

        const processedPredictions = result.raw_predictions.map(
          (value: number) => convertToDroughtLevel(value)
        );

        setPredictions(processedPredictions);
        setError(null); // Xóa thông báo lỗi trước đó
      } catch (err) {
        console.error("Lỗi chi tiết:", err);
        setError(
          err instanceof Error
            ? `Lỗi: ${err.message}`
            : "Đã xảy ra lỗi khi dự đoán"
        );
      } finally {
        setIsLoading(false);
      }
    };

    fetchPredictions();
  }, [
    coordinates,
    soilData,
    rawCsvData,
    apiEndpoint,
    selectedRegion,
    startDate,
    endDate,
  ]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
        <span className="ml-2 text-gray-500">Đang tạo dự đoán hạn hán...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Dự Đoán Hạn Hán</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-md">
            {error}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!predictions) return null;

  // GeoJSON style function based on the current week's prediction
  const getGeoJSONStyle = (weekIndex: number) => {
    return {
      fillColor:
        DROUGHT_COLORS[predictions[weekIndex] as keyof typeof DROUGHT_COLORS],
      weight: 2,
      opacity: 1,
      color: "white",
      fillOpacity: 0.8,
    };
  };

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>
          Dự Đoán Hạn Hán 6 Tuần Tiếp Theo
          {countyName && selectedRegion === "US" && (
            <span className="block text-sm text-gray-500 font-normal mt-1">
              Hạt: {countyName}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-4">
            {weekDates.map((week) => (
              <TabsTrigger key={week.id} value={week.id}>
                {week.label} ({week.date})
              </TabsTrigger>
            ))}
          </TabsList>

          {weekDates.map((week, index) => (
            <TabsContent key={week.id} value={week.id} className="space-y-4">
              <div className="font-medium">Dự đoán cho {week.date}:</div>
              <div
                className="mt-2 p-3 rounded-md font-bold text-lg"
                style={{
                  backgroundColor:
                    DROUGHT_COLORS[
                      predictions[index] as keyof typeof DROUGHT_COLORS
                    ],
                }}
              >
                {
                  DROUGHT_LABELS[
                    predictions[index] as keyof typeof DROUGHT_LABELS
                  ]
                }
              </div>

              <div className="h-[400px] rounded-md overflow-hidden border">
                <MapContainer
                  center={[coordinates.latitude!, coordinates.longitude!]}
                  zoom={selectedRegion === "US" ? 8 : 13}
                  style={{ height: "100%", width: "100%" }}
                >
                  <TileLayer
                    url="https://api.maptiler.com/maps/basic-v2/256/{z}/{x}/{y}.png?key=P96WEPQUms3wpDYVjOhc"
                    attribution='&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; OpenStreetMap contributors'
                  />

                  {/* Render county boundaries for US regions */}
                  {selectedRegion === "US" && countyData ? (
                    <GeoJSON
                      data={countyData}
                      style={() => getGeoJSONStyle(index)}
                    >
                      <Tooltip direction="center">
                        <div className="text-center">
                          <div className="font-bold">
                            {countyName ? `${countyName} County` : "County"}
                          </div>
                          <div>
                            {
                              DROUGHT_LABELS[
                                predictions[
                                  index
                                ] as keyof typeof DROUGHT_LABELS
                              ]
                            }
                          </div>
                          <div>({week.date})</div>
                        </div>
                      </Tooltip>
                      <Popup>
                        <div>
                          <strong>Hạt:</strong> {countyName || "N/A"}
                          <br />
                          <strong>FIPS:</strong> {countyFips || "N/A"}
                          <br />
                          <strong>Vị trí:</strong>{" "}
                          {coordinates.latitude?.toFixed(6)},{" "}
                          {coordinates.longitude?.toFixed(6)}
                          <br />
                          <strong>Dự đoán:</strong>{" "}
                          {
                            DROUGHT_LABELS[
                              predictions[index] as keyof typeof DROUGHT_LABELS
                            ]
                          }
                          <br />
                          <strong>Ngày:</strong> {week.date}
                        </div>
                      </Popup>
                    </GeoJSON>
                  ) : (
                    /* Fallback to circle if not US or county data not available */
                    <Circle
                      center={[coordinates.latitude!, coordinates.longitude!]}
                      radius={5000}
                      pathOptions={{
                        fillColor:
                          DROUGHT_COLORS[
                            predictions[index] as keyof typeof DROUGHT_COLORS
                          ],
                        fillOpacity: 0.8,
                        color: "white",
                        weight: 1,
                      }}
                    >
                      <Tooltip direction="center" permanent>
                        <div className="text-center">
                          <div className="font-bold">
                            {
                              DROUGHT_LABELS[
                                predictions[
                                  index
                                ] as keyof typeof DROUGHT_LABELS
                              ]
                            }
                          </div>
                          <div>({week.date})</div>
                        </div>
                      </Tooltip>
                      <Popup>
                        <div>
                          <strong>Vị trí:</strong>{" "}
                          {coordinates.latitude?.toFixed(6)},{" "}
                          {coordinates.longitude?.toFixed(6)}
                          <br />
                          <strong>Dự đoán:</strong>{" "}
                          {
                            DROUGHT_LABELS[
                              predictions[index] as keyof typeof DROUGHT_LABELS
                            ]
                          }
                          <br />
                          <strong>Ngày:</strong> {week.date}
                        </div>
                      </Popup>
                    </Circle>
                  )}
                </MapContainer>
              </div>
            </TabsContent>
          ))}
        </Tabs>

        {/* Thêm phần chú thích (legend) dưới biểu đồ */}
        <div className="flex flex-wrap justify-center mt-4 pt-4 border-t border-gray-200">
          <div className="grid grid-cols-3 gap-x-8 w-full">
            {/* Hàng 1 */}
            <div className="flex items-center">
              <div
                className="w-4 h-4 rounded-sm mr-2"
                style={{ backgroundColor: "rgba(166, 249, 166, 0.6)" }}
              ></div>
              <span className="text-sm text-gray-700">Không có hạn hán</span>
            </div>
            <div className="flex items-center">
              <div
                className="w-4 h-4 rounded-sm mr-2"
                style={{ backgroundColor: "rgba(255, 255, 190, 0.6)" }}
              ></div>
              <span className="text-sm text-gray-700">Khô hạn nhẹ (D0)</span>
            </div>
            <div className="flex items-center">
              <div
                className="w-4 h-4 rounded-sm mr-2"
                style={{ backgroundColor: "rgba(255, 211, 127, 0.6)" }}
              ></div>
              <span className="text-sm text-gray-700">Hạn trung bình (D1)</span>
            </div>

            {/* Hàng 2 */}
            <div className="flex items-center">
              <div
                className="w-4 h-4 rounded-sm mr-2"
                style={{ backgroundColor: "rgba(255, 170, 0, 0.6)" }}
              ></div>
              <span className="text-sm text-gray-700">
                Hạn nghiêm trọng (D2)
              </span>
            </div>
            <div className="flex items-center">
              <div
                className="w-4 h-4 rounded-sm mr-2"
                style={{ backgroundColor: "rgba(230, 0, 0, 0.6)" }}
              ></div>
              <span className="text-sm text-gray-700">
                Hạn cực kỳ nghiêm trọng (D3)
              </span>
            </div>
            <div className="flex items-center">
              <div
                className="w-4 h-4 rounded-sm mr-2"
                style={{ backgroundColor: "rgba(115, 0, 0, 0.6)" }}
              ></div>
              <span className="text-sm text-gray-700">Hạn thảm khốc (D4)</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
