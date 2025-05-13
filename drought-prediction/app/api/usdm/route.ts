// app/api/usdm/route.ts
import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  try {
    // Lấy các tham số từ URL
    const searchParams = request.nextUrl.searchParams;
    const aoi = searchParams.get("aoi");
    const startdate = searchParams.get("startdate");
    const enddate = searchParams.get("enddate");

    // Kiểm tra tham số bắt buộc
    if (!aoi || !startdate || !enddate) {
      return NextResponse.json(
        { error: "Thiếu tham số bắt buộc" },
        { status: 400 }
      );
    }

    // Xây dựng URL cho USDM API
    const usdmUrl = `https://usdmdataservices.unl.edu/api/CountyStatistics/GetDSCI?aoi=${aoi}&startdate=${startdate}&enddate=${enddate}&statisticsType=csv`;

    console.log(`Proxying request to: ${usdmUrl}`);

    // Gọi API USDM
    const response = await fetch(usdmUrl);

    if (!response.ok) {
      throw new Error(`USDM API error: ${response.status}`);
    }

    // Lấy dữ liệu CSV
    const csvData = await response.text();

    // Trả về dữ liệu CSV
    return new NextResponse(csvData, {
      headers: {
        "Content-Type": "text/csv",
      },
    });
  } catch (error) {
    console.error("Error fetching USDM data:", error);
    return NextResponse.json(
      {
        error: "Không thể lấy dữ liệu USDM",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

// Có thể thêm cấu hình CORS nếu cần thiết (thường không cần trong Next.js app router)
export const config = {
  runtime: "edge", // Tùy chọn: sử dụng runtime edge cho hiệu suất tốt hơn
};
