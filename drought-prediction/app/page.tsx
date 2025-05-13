"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { format, parse, isValid, subDays } from "date-fns";
import dynamic from "next/dynamic";
import { Loader2, Download, CalendarIcon, RefreshCw } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Calendar } from "@/components/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

// Dynamically import the MapComponent to avoid SSR issues with Leaflet
const MapComponent = dynamic(() => import("@/components/map-component"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-[500px] bg-gray-100">
      <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
      <span className="ml-2 text-gray-500">Đang tải bản đồ...</span>
    </div>
  ),
});

// Dynamically import the DroughtPredictionMaps component
const DroughtPredictionMaps = dynamic(
  () => import("@/components/drought-prediction-map"),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-[400px] bg-gray-100">
        <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
        <span className="ml-2 text-gray-500">Đang tải dự đoán hạn hán...</span>
      </div>
    ),
  }
);

// Model types configuration
const MODEL_TYPES = {
  LSTM: {
    label: "LSTM",
    apis: {
      GLOBAL: "http://localhost:7860/predict",
      US: "http://localhost:7861/predict",
    },
  },
  GRU: {
    label: "GRU",
    apis: {
      GLOBAL: "http://localhost:7862/predict",
      US: "http://localhost:7863/predict",
    },
  },
  TRANSFORMER: {
    label: "Transformer",
    apis: {
      GLOBAL: "http://localhost:7864/predict",
      US: "http://localhost:7865/predict",
    },
  },
  LSTMAttn: {
    label: "LSTM & Attention",
    apis: {
      GLOBAL: "http://localhost:7866/predict",
      US: "http://localhost:7867/predict",
    },
  },
};

// Region types
const REGION_TYPES = {
  GLOBAL: "Toàn Thế Giới",
  US: "Hoa Kỳ",
};

// Helper function to download CSV data
const downloadCSV = (csvContent, fileName) => {
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");

  // Create a URL for the blob
  const url = URL.createObjectURL(blob);
  link.setAttribute("href", url);
  link.setAttribute("download", fileName);
  link.style.visibility = "hidden";

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// Helper function to convert soil data to CSV
const convertSoilDataToCSV = (soilData) => {
  const columns = [
    "Elevation",
    "SlopesCl1",
    "SlopesCl2",
    "SlopesCl3",
    "SlopesCl4",
    "SlopesCl5",
    "SlopesCl6",
    "SlopesCl7",
    "SlopesCl8",
    "AspectClN",
    "AspectClE",
    "AspectClS",
    "AspectClW",
    "AspectClU",
    "WAT",
    "NVG",
    "URB",
    "GRS",
    "FOR",
    "CULTRF",
    "CULTIR",
    "CULT",
    "sq1",
    "sq2",
    "sq3",
    "sq4",
    "sq5",
    "sq6",
    "sq7",
    "LandMask",
  ];

  // Parse the soilData string into an array
  const valuesArray = JSON.parse(soilData);

  // Create CSV header
  let csvContent = columns.join(",") + "\n";

  // Add the values row
  csvContent += valuesArray.join(",");

  return csvContent;
};

// Validate coordinates function
const isValidCoordinate = (value) => {
  // Check if the value is a valid number
  if (value === null || value === undefined || isNaN(Number(value))) {
    return false;
  }

  // Convert to number to be sure
  const num = Number(value);

  // General validation for latitude (-90 to 90) and longitude (-180 to 180)
  // This is loose validation as we're not distinguishing between lat and long
  return num >= -180 && num <= 180;
};

// Parse a date string in format DD/MM/YYYY or YYYY-MM-DD
const parseDateInput = (dateStr) => {
  // Try to parse as DD/MM/YYYY
  let parsedDate = parse(dateStr, "dd/MM/yyyy", new Date());

  // If not valid, try YYYY-MM-DD format
  if (!isValid(parsedDate)) {
    parsedDate = parse(dateStr, "yyyy-MM-dd", new Date());
  }

  return isValid(parsedDate) ? parsedDate : null;
};

export default function Home() {
  // State for coordinates
  const [coordinates, setCoordinates] = useState({
    latitude: null,
    longitude: null,
  });

  // State for coordinate input fields
  const [latitudeInput, setLatitudeInput] = useState("");
  const [longitudeInput, setLongitudeInput] = useState("");

  // State for dates
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  // State for date input fields
  const [startDateInput, setStartDateInput] = useState("");
  const [endDateInput, setEndDateInput] = useState("");

  // State for calendar popover
  const [startDateOpen, setStartDateOpen] = useState(false);
  const [endDateOpen, setEndDateOpen] = useState(false);

  // State for region and model selections
  const [selectedRegion, setSelectedRegion] = useState("GLOBAL");
  const [selectedModel, setSelectedModel] = useState("LSTM");

  // State for API data
  const [soilData, setSoilData] = useState(null);
  const [timeSeriesData, setTimeSeriesData] = useState(null);
  const [rawCsvData, setRawCsvData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // State to control whether to show drought predictions
  const [showDroughtPredictions, setShowDroughtPredictions] = useState(false);

  // Initialize dates on component mount
  useEffect(() => {
    const endDateDefault = new Date("2025-05-10");
    const sixMonthsAgo = subDays(endDateDefault, 174);

    setStartDate(format(sixMonthsAgo, "yyyy-MM-dd"));
    setEndDate(format(endDateDefault, "yyyy-MM-dd"));

    // Set initial values for the input fields
    setStartDateInput(format(sixMonthsAgo, "dd/MM/yyyy"));
    setEndDateInput(format(endDateDefault, "dd/MM/yyyy"));
  }, []);

  // Update input fields when coordinates change (from map)
  useEffect(() => {
    if (coordinates.latitude !== null) {
      setLatitudeInput(coordinates.latitude.toString());
    }
    if (coordinates.longitude !== null) {
      setLongitudeInput(coordinates.longitude.toString());
    }
  }, [coordinates.latitude, coordinates.longitude]);

  // Update date input fields when dates change
  useEffect(() => {
    if (startDate) {
      setStartDateInput(format(new Date(startDate), "dd/MM/yyyy"));
    }
    if (endDate) {
      setEndDateInput(format(new Date(endDate), "dd/MM/yyyy"));
    }
  }, [startDate, endDate]);

  // Handle coordinate input changes
  const handleLatitudeChange = (e) => {
    const value = e.target.value;
    setLatitudeInput(value);

    // Update coordinates if valid
    if (isValidCoordinate(value)) {
      setCoordinates((prev) => ({
        ...prev,
        latitude: Number(value),
      }));
    }
  };

  const handleLongitudeChange = (e) => {
    const value = e.target.value;
    setLongitudeInput(value);

    // Update coordinates if valid
    if (isValidCoordinate(value)) {
      setCoordinates((prev) => ({
        ...prev,
        longitude: Number(value),
      }));
    }
  };

  // Handle direct date input changes
  const handleStartDateInputChange = (e) => {
    const value = e.target.value;
    setStartDateInput(value);
  };

  const handleEndDateInputChange = (e) => {
    const value = e.target.value;
    setEndDateInput(value);
  };

  // Apply date input changes
  const applyStartDateInput = () => {
    const parsedDate = parseDateInput(startDateInput);
    if (parsedDate) {
      setStartDate(format(parsedDate, "yyyy-MM-dd"));
    } else {
      // Reset to previous valid date if input is invalid
      if (startDate) {
        setStartDateInput(format(new Date(startDate), "dd/MM/yyyy"));
      }
    }
  };

  const applyEndDateInput = () => {
    const parsedDate = parseDateInput(endDateInput);
    if (parsedDate) {
      setEndDate(format(parsedDate, "yyyy-MM-dd"));
    } else {
      // Reset to previous valid date if input is invalid
      if (endDate) {
        setEndDateInput(format(new Date(endDate), "dd/MM/yyyy"));
      }
    }
  };

  // Handle date changes from the calendar
  const handleStartDateChange = (date) => {
    setStartDate(format(date, "yyyy-MM-dd"));
    setStartDateInput(format(date, "dd/MM/yyyy"));
    setStartDateOpen(false);
  };

  const handleEndDateChange = (date) => {
    setEndDate(format(date, "yyyy-MM-dd"));
    setEndDateInput(format(date, "dd/MM/yyyy"));
    setEndDateOpen(false);
  };

  // Get the current API endpoint based on selected region and model
  const getCurrentApiEndpoint = () => {
    return MODEL_TYPES[selectedModel].apis[selectedRegion];
  };

  // Fetch data from APIs
  const fetchData = async () => {
    if (!coordinates.latitude || !coordinates.longitude) {
      setError("Vui lòng chọn một vị trí trên bản đồ trước");
      return false; // Return false to indicate failure
    }

    // Validate dates
    const startDateObj = new Date(startDate);
    const endDateObj = new Date(endDate);

    // Check if start date is after end date
    if (startDateObj > endDateObj) {
      setError("Ngày bắt đầu phải trước ngày kết thúc");
      return false; // Return false to indicate failure
    }

    // Check date range (maximum 175 days)
    const differenceInDays = Math.ceil(
      (endDateObj.getTime() - startDateObj.getTime()) / (1000 * 3600 * 24)
    );

    if (differenceInDays > 174) {
      setError("Khoảng thời gian không được vượt quá 175 ngày");
      return false; // Return false to indicate failure
    }

    setIsLoading(true);
    setError(null);
    setShowDroughtPredictions(false);

    try {
      // Format dates for API request (remove hyphens)
      const formattedStartDate = startDate.replace(/-/g, "");
      const formattedEndDate = endDate.replace(/-/g, "");

      // Fetch soil data
      const soilResponse = await fetch("/api/soil-data", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          latitude: coordinates.latitude,
          longitude: coordinates.longitude,
        }),
      });

      if (!soilResponse.ok) {
        throw new Error("Không thể lấy dữ liệu đất");
      }

      const soilResult = await soilResponse.json();

      // Store the values in X_static
      const X_static = soilResult.values;
      setSoilData(X_static);

      // Check if last value of X_static is 0
      const valuesArray = JSON.parse(X_static);
      if (valuesArray[0] === 0 && valuesArray[valuesArray.length - 1] === 0) {
        setError("Bạn đang không ở khu vực có đất");
        setIsLoading(false);
        return false; // Return false to indicate that we should stop here
      }

      // Fetch time series data
      const timeSeriesResponse = await fetch("/api/time-series", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          latitude: coordinates.latitude,
          longitude: coordinates.longitude,
          start: formattedStartDate,
          end: formattedEndDate,
        }),
      });

      if (!timeSeriesResponse.ok) {
        throw new Error("Không thể lấy dữ liệu chuỗi thời gian");
      }

      const timeSeriesResult = await timeSeriesResponse.json();
      setTimeSeriesData(timeSeriesResult.parsedData);
      setRawCsvData(timeSeriesResult.rawCsv);

      // Enable drought predictions if we have both soil data and time series data
      if (X_static && timeSeriesResult.rawCsv) {
        setShowDroughtPredictions(true);
      }

      return true; // Return true to indicate success
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Đã xảy ra lỗi không xác định"
      );
      return false; // Return false to indicate failure
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-fetch data when coordinates or dates change
  useEffect(() => {
    let isMounted = true; // Flag to track if component is mounted

    const autoFetchData = async () => {
      if (
        coordinates.latitude &&
        coordinates.longitude &&
        startDate &&
        endDate
      ) {
        // Only fetch data if the component is still mounted
        if (isMounted) {
          const result = await fetchData();
          // If fetchData returns false, we've encountered an error and should stop processing
          if (!result) {
            console.log("Fetching data stopped due to validation error");
            // Clear results data when there's an error
            if (error === "Bạn đang không ở khu vực có đất" || error) {
              setSoilData(null);
              setTimeSeriesData(null);
              setRawCsvData(null);
              setShowDroughtPredictions(false);
            }
          }
        }
      }
    };

    autoFetchData();

    // Cleanup function
    return () => {
      isMounted = false; // Set flag to false when component unmounts
    };
  }, [coordinates.latitude, coordinates.longitude, startDate, endDate]);

  // Reset date range to last 6 months
  const resetDateRange = () => {
    const now = new Date();
    const sixMonthsAgo = subDays(now, 174);

    setStartDate(format(sixMonthsAgo, "yyyy-MM-dd"));
    setEndDate(format(now, "yyyy-MM-dd"));

    setStartDateInput(format(sixMonthsAgo, "dd/MM/yyyy"));
    setEndDateInput(format(now, "dd/MM/yyyy"));
  };

  // Handle coordinate selection from map
  const handleCoordinateSelect = (lat, lng) => {
    setCoordinates({
      latitude: lat,
      longitude: lng,
    });
  };

  // Handle region change
  const handleRegionChange = (value) => {
    setSelectedRegion(value);
    // Reset drought predictions when region changes
    setShowDroughtPredictions(false);
  };

  // Handle model change
  const handleModelChange = (value) => {
    setSelectedModel(value);
    // Reset drought predictions when model changes
    setShowDroughtPredictions(false);
  };

  // Auto-fetch data when coordinates or dates change
  useEffect(() => {
    if (coordinates.latitude && coordinates.longitude && startDate && endDate) {
      fetchData();
    }
  }, [coordinates.latitude, coordinates.longitude, startDate, endDate]);

  // Function to download soil data as CSV
  const handleDownloadSoilData = () => {
    if (soilData) {
      const csvContent = convertSoilDataToCSV(soilData);
      downloadCSV(
        csvContent,
        `soil-data-${coordinates.latitude}-${coordinates.longitude}.csv`
      );
    }
  };

  // Create CSV table component
  const CsvTable = ({ data }) => {
    if (!data || data.length === 0) return <p>Không có dữ liệu</p>;

    // Get headers from the first item
    const headers = Object.keys(data[0]);

    return (
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {headers.map((header, index) => (
                <th
                  key={index}
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data.slice(0, 20).map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className={rowIndex % 2 === 0 ? "bg-white" : "bg-gray-50"}
              >
                {headers.map((header, colIndex) => (
                  <td
                    key={`${rowIndex}-${colIndex}`}
                    className="px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                  >
                    {row[header] !== null && row[header] !== undefined
                      ? String(row[header])
                      : "-"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {data.length > 20 && (
          <p className="text-sm text-gray-500 mt-2 pl-6">
            Hiển thị 20/{data.length} dòng. Tải xuống CSV để xem toàn bộ dữ
            liệu.
          </p>
        )}
      </div>
    );
  };

  // Create soil data table component
  const SoilDataTable = ({ data }) => {
    if (!data) return <p>Không có dữ liệu</p>;

    // Parse the data string into an array
    const valuesArray = JSON.parse(data);
    const columns = [
      "Elevation",
      "SlopesCl1",
      "SlopesCl2",
      "SlopesCl3",
      "SlopesCl4",
      "SlopesCl5",
      "SlopesCl6",
      "SlopesCl7",
      "SlopesCl8",
      "AspectClN",
      "AspectClE",
      "AspectClS",
      "AspectClW",
      "AspectClU",
      "WAT",
      "NVG",
      "URB",
      "GRS",
      "FOR",
      "CULTRF",
      "CULTIR",
      "CULT",
      "sq1",
      "sq2",
      "sq3",
      "sq4",
      "sq5",
      "sq6",
      "sq7",
      "LandMask",
    ];

    return (
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {columns.map((column, index) => (
                <th
                  key={index}
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            <tr className="bg-white">
              {valuesArray.map((value, index) => (
                <td
                  key={index}
                  className="px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                >
                  {value !== null && value !== undefined ? String(value) : "-"}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="container mx-auto py-6 space-y-6">
      <h1 className="text-2xl font-bold text-center mb-4 flex items-center justify-center space-x-2">
        <img src="/data/earth-globe.png" alt="Logo" className="w-8 h-8" />
        <span>Trình Khám Phá Dữ Liệu Bản Đồ Tương Tác & Dự Đoán Hạn Hán</span>
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Map Card */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Bản Đồ</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[500px] rounded-md overflow-hidden">
              <MapComponent
                onCoordinateSelect={handleCoordinateSelect}
                coordinates={coordinates}
              />
            </div>
          </CardContent>
        </Card>

        {/* Location Information Card */}
        <Card>
          <CardHeader>
            <CardTitle>Thông Tin Vị Trí</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="latitude">Vĩ Độ</Label>
              <Input
                id="latitude"
                value={latitudeInput}
                onChange={(e) => setLatitudeInput(e.target.value)} // Cập nhật state nhưng chưa xử lý
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleLatitudeChange(e); // Gọi xử lý khi Enter
                  }
                }}
                placeholder="Nhập vĩ độ hoặc chọn trên bản đồ"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="longitude">Kinh Độ</Label>
              <Input
                id="longitude"
                value={longitudeInput}
                onChange={(e) => setLongitudeInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleLongitudeChange(e); // Gọi hàm khi nhấn Enter
                  }
                }}
                placeholder="Nhập kinh độ hoặc chọn trên bản đồ"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Khoảng Thời Gian</Label>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 p-0"
                  onClick={resetDateRange}
                  title="Đặt lại khoảng thời gian về 6 tháng gần đây"
                >
                  <RefreshCw className="h-4 w-4" />
                </Button>
              </div>

              <div className="flex flex-col space-y-3">
                {/* Calendar and Input for Start Date */}
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="startDate" className="text-right col-span-1">
                    Từ:
                  </Label>
                  <div className="col-span-3 flex">
                    <Input
                      id="startDateInput"
                      value={startDateInput}
                      onChange={handleStartDateInputChange}
                      onBlur={applyStartDateInput}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          applyStartDateInput();
                        }
                      }}
                      placeholder="DD/MM/YYYY"
                      className="rounded-r-none"
                    />
                    <Popover
                      open={startDateOpen}
                      onOpenChange={setStartDateOpen}
                    >
                      <PopoverTrigger asChild>
                        <Button
                          variant="outline"
                          className="rounded-l-none border-l-0"
                        >
                          <CalendarIcon className="h-4 w-4" />
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-auto p-0" align="end">
                        <Calendar
                          mode="single"
                          selected={startDate ? new Date(startDate) : undefined}
                          defaultMonth={
                            startDate ? new Date(startDate) : undefined
                          }
                          onSelect={handleStartDateChange}
                          fromDate={new Date("2000-01-01")}
                          toDate={new Date("2025-05-10")}
                          initialFocus
                        />
                      </PopoverContent>
                    </Popover>
                  </div>
                </div>

                {/* Calendar and Input for End Date */}
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="endDate" className="text-right col-span-1">
                    Đến:
                  </Label>
                  <div className="col-span-3 flex">
                    <Input
                      id="endDateInput"
                      value={endDateInput}
                      onChange={handleEndDateInputChange}
                      onBlur={applyEndDateInput}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          applyEndDateInput();
                        }
                      }}
                      placeholder="DD/MM/YYYY"
                      className="rounded-r-none"
                    />
                    <Popover open={endDateOpen} onOpenChange={setEndDateOpen}>
                      <PopoverTrigger asChild>
                        <Button
                          variant="outline"
                          className="rounded-l-none border-l-0"
                        >
                          <CalendarIcon className="h-4 w-4" />
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-auto p-0" align="end">
                        <Calendar
                          mode="single"
                          selected={endDate ? new Date(endDate) : undefined}
                          defaultMonth={endDate ? new Date(endDate) : undefined}
                          onSelect={handleEndDateChange}
                          fromDate={new Date("2000-01-01")}
                          toDate={new Date("2025-05-10")}
                          initialFocus
                        />
                      </PopoverContent>
                    </Popover>
                  </div>
                </div>
              </div>
            </div>

            {/* Region Selection */}
            <div className="space-y-2">
              <Label htmlFor="region">Khu Vực</Label>
              <Select value={selectedRegion} onValueChange={handleRegionChange}>
                <SelectTrigger id="region">
                  <SelectValue placeholder="Chọn khu vực" />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(REGION_TYPES).map(([key, value]) => (
                    <SelectItem key={key} value={key}>
                      {value}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Model Selection */}
            <div className="space-y-2">
              <Label htmlFor="model">Mô Hình</Label>
              <Select value={selectedModel} onValueChange={handleModelChange}>
                <SelectTrigger id="model">
                  <SelectValue placeholder="Chọn mô hình" />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(MODEL_TYPES).map(([key, value]) => (
                    <SelectItem key={key} value={key}>
                      {value.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Button
              className="w-full"
              onClick={fetchData}
              disabled={
                isLoading || !coordinates.latitude || !coordinates.longitude
              }
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Đang Lấy Dữ Liệu...
                </>
              ) : (
                "Lấy Dữ Liệu"
              )}
            </Button>
            {error && (
              <div
                className={`text-sm mt-2 ${
                  error === "Bạn đang không ở khu vực có đất"
                    ? "text-orange-500 font-semibold"
                    : "text-red-500"
                }`}
              >
                {error}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Results Section */}
      {(soilData || timeSeriesData) &&
        !(error && error === "Bạn đang không ở khu vực có đất") && (
          <Card>
            <CardHeader>
              <CardTitle className="flex justify-between items-center">
                <span>Kết Quả</span>
                <div className="flex gap-2">
                  {soilData && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleDownloadSoilData}
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Tải xuống CSV Đất
                    </Button>
                  )}
                  {rawCsvData && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        downloadCSV(
                          rawCsvData,
                          `weather-data-${coordinates.latitude}-${coordinates.longitude}.csv`
                        )
                      }
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Tải xuống CSV Thời Tiết
                    </Button>
                  )}
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="soil">
                <TabsList className="mb-4">
                  <TabsTrigger value="soil">Dữ Liệu Đất</TabsTrigger>
                  <TabsTrigger value="timeSeries">
                    Dữ Liệu Chuỗi Thời Gian
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="soil" className="space-y-4">
                  <h3 className="font-medium">Dữ Liệu Đất:</h3>
                  {soilData ? (
                    <div className="bg-white rounded-md border max-h-[600px] overflow-auto">
                      <SoilDataTable data={soilData} />
                    </div>
                  ) : (
                    <p>Không có dữ liệu</p>
                  )}
                </TabsContent>

                <TabsContent value="timeSeries">
                  <h3 className="font-medium">Dữ Liệu Chuỗi Thời Gian:</h3>
                  {timeSeriesData ? (
                    <div className="bg-white rounded-md border max-h-[600px] overflow-auto">
                      <CsvTable data={timeSeriesData} />
                    </div>
                  ) : (
                    <p>Không có dữ liệu</p>
                  )}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        )}

      {/* Drought Prediction Maps Section */}
      {showDroughtPredictions && (
        <DroughtPredictionMaps
          coordinates={coordinates}
          soilData={soilData}
          rawCsvData={rawCsvData}
          endDate={endDate}
          startDate={startDate}
          selectedRegion={selectedRegion}
          selectedModel={selectedModel}
          apiEndpoint={getCurrentApiEndpoint()}
        />
      )}
    </div>
  );
}
