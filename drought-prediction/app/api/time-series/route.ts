import { NextResponse } from "next/server";
import Papa from "papaparse";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { latitude, longitude, start, end } = body;

    // Validate input
    if (!latitude || !longitude || !start || !end) {
      return NextResponse.json(
        { error: "Vĩ độ, kinh độ, ngày bắt đầu và ngày kết thúc là bắt buộc" },
        { status: 400 }
      );
    }

    // Prepare parameters for NASA POWER API
    const params = new URLSearchParams({
      parameters:
        "PRECTOT,PS,QV2M,T2M,T2MDEW,T2MWET,T2M_MAX,T2M_MIN,T2M_RANGE,TS,WS10M,WS10M_MAX,WS10M_MIN,WS10M_RANGE,WS50M,WS50M_MAX,WS50M_MIN,WS50M_RANGE",
      community: "AG",
      longitude: longitude.toString(),
      latitude: latitude.toString(),
      start: start,
      end: end,
      format: "CSV", // Changed from JSON to CSV
    });

    // Call the NASA POWER API
    const url = `https://power.larc.nasa.gov/api/temporal/daily/point?${params.toString()}`;

    const response = await fetch(url);

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Lỗi API NASA POWER:", errorText);
      return NextResponse.json(
        { error: "Không thể lấy dữ liệu chuỗi thời gian" },
        { status: response.status }
      );
    }

    // Get raw CSV data
    const csvText = await response.text();

    // Remove the first 26 lines (NASA metadata) from CSV
    const csvLines = csvText.split("\n");
    const dataLinesOnly = csvLines.slice(26).join("\n");

    // Parse CSV to JSON using Papaparse
    const parsedData = Papa.parse(dataLinesOnly, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
    });

    // Return both the raw CSV and parsed data
    return NextResponse.json({
      rawCsv: csvText,
      parsedData: parsedData.data,
      meta: parsedData.meta,
    });
  } catch (error) {
    console.error("Lỗi API chuỗi thời gian:", error);
    return NextResponse.json({ error: "Lỗi máy chủ nội bộ" }, { status: 500 });
  }
}
