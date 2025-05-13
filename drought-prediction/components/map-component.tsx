"use client";

import type React from "react";
import { useState, useRef, useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  useMapEvents,
  Popup,
  useMap,
} from "react-leaflet";
import { Icon } from "leaflet";
import "leaflet/dist/leaflet.css";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search } from "lucide-react";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Polygon, Tooltip } from "react-leaflet";

// Define the marker icon
const customIcon = new Icon({
  iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

// Component to update map view when coordinates change
function MapUpdater({ position }: { position: [number, number] }) {
  const map = useMap();

  useEffect(() => {
    if (position) {
      map.setView(position, 14); 
    }
  }, [map, position]);
  return null;
}

// Component to handle map clicks and events
function MapEvents({
  onCoordinateSelect,
}: {
  onCoordinateSelect: (lat: number, lng: number) => void;
}) {
  const map = useMapEvents({
    click: (e) => {
      const { lat, lng } = e.latlng;
      onCoordinateSelect(
        Number.parseFloat(lat.toFixed(6)),
        Number.parseFloat(lng.toFixed(6))
      );
    },
  });

  return null;
}

// CSS styles for z-index control
const styles = {
  popoverContentStyle: {
    zIndex: 1000, // Ensure the popover content is above the map
  },
  mapContainerStyle: {
    height: "100%",
    width: "100%",
    position: "relative",
    zIndex: 1, // Lower z-index for the map
  },
};

interface MapComponentProps {
  onCoordinateSelect: (lat: number, lng: number) => void;
  coordinates: {
    latitude: number | null;
    longitude: number | null;
  };
}

interface SearchResult {
  place_id: number;
  display_name: string;
  lat: string;
  lon: string;
  boundingbox?: string[];
  importance?: number;
  osm_type?: string;
  osm_id?: number;
  type?: string;
  class?: string;
}

export default function MapComponent({
  onCoordinateSelect,
  coordinates,
}: MapComponentProps) {
  const [position, setPosition] = useState<[number, number] | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isPopoverOpen, setIsPopoverOpen] = useState<boolean>(false);
  const mapRef = useRef<any>(null);

  // Set initial position based on user's location or default to Vietnam
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setPosition([latitude, longitude]);
        },
        () => {
          // Default position if geolocation is denied (Hanoi, Vietnam)
          setPosition([21.0285, 105.8542]);
        }
      );
    } else {
      // Default position if geolocation is not supported
      setPosition([21.0285, 105.8542]);
    }
  }, []);

  // Update position when coordinates prop changes
  useEffect(() => {
    if (coordinates.latitude !== null && coordinates.longitude !== null) {
      setPosition([coordinates.latitude, coordinates.longitude]);
    }
  }, [coordinates.latitude, coordinates.longitude]);

  // Handle marker position change
  const handleMarkerPositionChange = (lat: number, lng: number) => {
    setPosition([lat, lng]);
    onCoordinateSelect(lat, lng);
  };

  // Function to get detailed information about a location by its ID
  const getLocationDetail = async (osmType: string, osmId: number) => {
    try {
      // Convert OSM type to the format required by the API
      const osmTypeFormatted =
        osmType === "node"
          ? "N"
          : osmType === "way"
          ? "W"
          : osmType === "relation"
          ? "R"
          : "";

      if (!osmTypeFormatted) return null;

      const response = await fetch(
        `https://nominatim.openstreetmap.org/details?osmtype=${osmTypeFormatted}&osmid=${osmId}&format=json&addressdetails=1&extratags=1&namedetails=1`
      );

      if (!response.ok) {
        throw new Error("Không thể lấy chi tiết địa điểm");
      }

      return await response.json();
    } catch (error) {
      console.error("Lỗi khi lấy chi tiết địa điểm:", error);
      return null;
    }
  };

  // Search for locations as user types
  useEffect(() => {
    const searchLocations = async () => {
      if (searchQuery.length < 2) {
        setSearchResults([]);
        return;
      }

      setIsSearching(true);
      try {
        // Using Nominatim for geocoding with Vietnamese language preference
        const response = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
            searchQuery
          )}&accept-language=vi&limit=5&addressdetails=1`
        );

        if (!response.ok) {
          throw new Error("Tìm kiếm thất bại");
        }

        const data = await response.json();
        setSearchResults(data);
        setIsPopoverOpen(data.length > 0);
      } catch (error) {
        console.error("Lỗi tìm kiếm:", error);
      } finally {
        setIsSearching(false);
      }
    };

    const debounce = setTimeout(() => {
      if (searchQuery.length >= 2) {
        searchLocations();
      }
    }, 300);

    return () => clearTimeout(debounce);
  }, [searchQuery]);

  // Handle location selection from search results
  const handleLocationSelect = async (result: SearchResult) => {
    const latitude = Number.parseFloat(result.lat);
    const longitude = Number.parseFloat(result.lon);

    // Get more details about the location if osm_type and osm_id are available
    if (result.osm_type && result.osm_id) {
      const details = await getLocationDetail(result.osm_type, result.osm_id);

      if (details && details.centroid) {
        // Use the centroid coordinates if available
        const centroidLat = Number.parseFloat(details.centroid.coordinates[1]);
        const centroidLon = Number.parseFloat(details.centroid.coordinates[0]);

        // Update marker position with centroid
        setPosition([centroidLat, centroidLon]);
        onCoordinateSelect(centroidLat, centroidLon);
      } else {
        // Fallback to regular coordinates
        setPosition([latitude, longitude]);
        onCoordinateSelect(latitude, longitude);
      }
    } else {
      // If osm_type or osm_id not available, use regular coordinates
      setPosition([latitude, longitude]);
      onCoordinateSelect(latitude, longitude);
    }

    // Close popover and update search field
    setIsPopoverOpen(false);
    setSearchQuery(result.display_name);
  };

  // Handle search form submission
  const handleSearchSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (searchQuery.trim() === "") return;

    if (searchResults.length > 0) {
      const firstResult = searchResults[0];
      const latitude = Number.parseFloat(firstResult.lat);
      const longitude = Number.parseFloat(firstResult.lon);

      // Nếu có thông tin chi tiết, ưu tiên centroid
      if (firstResult.osm_type && firstResult.osm_id) {
        const details = await getLocationDetail(
          firstResult.osm_type,
          firstResult.osm_id
        );
        if (details?.centroid) {
          const centroidLat = Number.parseFloat(
            details.centroid.coordinates[1]
          );
          const centroidLon = Number.parseFloat(
            details.centroid.coordinates[0]
          );
          setPosition([centroidLat, centroidLon]);
          onCoordinateSelect(centroidLat, centroidLon);
          mapRef.current?.setView([centroidLat, centroidLon], 14); // 👈 Zoom 14 khi nhấn Enter
          return;
        }
      }

      // Nếu không có centroid, dùng lat/lon gốc
      setPosition([latitude, longitude]);
      onCoordinateSelect(latitude, longitude);
      mapRef.current?.setView([latitude, longitude], 14); // 👈 Zoom 14 khi nhấn Enter
      setSearchQuery(firstResult.display_name);
    } else {
      alert("Không tìm thấy địa điểm phù hợp.");
    }
  };
  

  if (!position) {
    return <div>Đang tải bản đồ...</div>;
  }

  return (
    <div className="h-full flex flex-col">
      <form
        onSubmit={handleSearchSubmit}
        className="flex gap-2 mb-2 relative z-50"
      >
        <div className="relative flex-1">
          <Popover open={isPopoverOpen} onOpenChange={setIsPopoverOpen}>
            <PopoverTrigger asChild>
              <div className="w-full">
                <Input
                  type="text"
                  placeholder="Tìm kiếm địa điểm..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full"
                />
              </div>
            </PopoverTrigger>
            <PopoverContent
              className="p-0 w-[300px] lg:w-[400px]"
              align="start"
              style={styles.popoverContentStyle}
            >
              <Command>
                <CommandList>
                  <CommandEmpty>Không tìm thấy kết quả</CommandEmpty>
                  <CommandGroup heading="Kết quả tìm kiếm">
                    {searchResults.map((result) => (
                      <CommandItem
                        key={result.place_id}
                        onSelect={() => handleLocationSelect(result)}
                        className="cursor-pointer"
                      >
                        <div className="flex flex-col">
                          <span className="font-medium">
                            {result.display_name.split(", ")[0]}
                          </span>
                          <span className="text-xs text-muted-foreground truncate">
                            {result.display_name}
                          </span>
                        </div>
                      </CommandItem>
                    ))}
                  </CommandGroup>
                </CommandList>
              </Command>
            </PopoverContent>
          </Popover>
        </div>
        <Button type="submit" disabled={isSearching}>
          {isSearching ? "Đang tìm..." : <Search className="h-4 w-4" />}
        </Button>
      </form>

      <div className="flex-1 relative z-10">
        <MapContainer
          center={position}
          zoom={13}
          style={styles.mapContainerStyle}
          ref={mapRef}
        >
          {/* Using a Vietnamese-friendly tile layer */}
          <TileLayer
            url="https://api.maptiler.com/maps/basic-v2/256/{z}/{x}/{y}.png?key=P96WEPQUms3wpDYVjOhc"
            attribution='&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; OpenStreetMap contributors'
          />

          {position && (
            <Marker
              position={position}
              icon={customIcon}
              draggable={true}
              eventHandlers={{
                dragend: (e) => {
                  const marker = e.target;
                  const position = marker.getLatLng();
                  handleMarkerPositionChange(
                    Number.parseFloat(position.lat.toFixed(6)),
                    Number.parseFloat(position.lng.toFixed(6))
                  );
                },
              }}
            >
              <Popup>
                Vĩ độ: {position[0].toFixed(6)}
                <br />
                Kinh độ: {position[1].toFixed(6)}
              </Popup>
            </Marker>
          )}

          <MapEvents onCoordinateSelect={handleMarkerPositionChange} />
          <MapUpdater position={position} />
        </MapContainer>
      </div>
    </div>
  );
}
