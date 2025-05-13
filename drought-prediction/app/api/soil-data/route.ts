import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { latitude, longitude } = body;

    // Validate input
    if (latitude === undefined || longitude === undefined) {
      return NextResponse.json(
        { error: "Vĩ độ và kinh độ là bắt buộc" },
        { status: 400 }
      );
    }

    // Call the soil data API
    const response = await fetch(
      "https://l1af2027-soil-data.hf.space/get_data",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ latitude, longitude }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Lỗi API đất:", errorText);
      return NextResponse.json(
        { error: "Không thể lấy dữ liệu đất" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Lỗi API dữ liệu đất:", error);
    return NextResponse.json({ error: "Lỗi máy chủ nội bộ" }, { status: 500 });
  }
}
