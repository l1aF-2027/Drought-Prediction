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
  LayersControl,
} from "react-leaflet";
import { Icon } from "leaflet";
import "leaflet/dist/leaflet.css";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, Loader2 } from "lucide-react";
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

// Define available map layers
const mapLayers = {
  basic: {
    name: "Bản đồ cơ bản",
    url: "https://api.maptiler.com/maps/basic-v2/256/{z}/{x}/{y}.png?key=P96WEPQUms3wpDYVjOhc",
    attribution:
      '&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; OpenStreetMap contributors',
  },
  satellite: {
    name: "Vệ tinh",
    url: "https://api.maptiler.com/maps/satellite/256/{z}/{x}/{y}.jpg?key=P96WEPQUms3wpDYVjOhc",
    attribution:
      '&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; OpenStreetMap contributors',
  },
  hybrid: {
    name: "Vệ tinh",
    url: "https://api.maptiler.com/maps/hybrid/256/{z}/{x}/{y}.jpg?key=P96WEPQUms3wpDYVjOhc",
    attribution:
      '&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; OpenStreetMap contributors',
  },
  streets: {
    name: "Đường phố",
    url: "https://api.maptiler.com/maps/streets-v2/256/{z}/{x}/{y}.png?key=P96WEPQUms3wpDYVjOhc",
    attribution:
      '&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; OpenStreetMap contributors',
  },
  openStreetMap: {
    name: "OpenStreetMap",
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  },
};

// Function to normalize coordinates to standard range
function normalizeCoordinates(lat, lng) {
  // Normalize latitude to range -90 to 90
  lat = Math.max(-90, Math.min(90, lat));

  // Normalize longitude to range -180 to 180
  lng = ((lng + 540) % 360) - 180;

  return [lat, lng];
}

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
      // Normalize coordinates before passing them to the handler
      const [normalizedLat, normalizedLng] = normalizeCoordinates(lat, lng);
      onCoordinateSelect(
        Number.parseFloat(normalizedLat.toFixed(6)),
        Number.parseFloat(normalizedLng.toFixed(6))
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
  const searchInputRef = useRef<HTMLInputElement>(null);
  const [currentGpsPosition, setCurrentGpsPosition] = useState<
    [number, number] | null
  >(null);

  // Get user's current location and set it
  const getCurrentPosition = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          // Normalize coordinates
          const [normalizedLat, normalizedLng] = normalizeCoordinates(
            latitude,
            longitude
          );
          const newPosition: [number, number] = [normalizedLat, normalizedLng];
          setCurrentGpsPosition(newPosition);
          setPosition(newPosition);
          // Also update the parent component with the GPS position
          onCoordinateSelect(normalizedLat, normalizedLng);
        },
        (error) => {
          console.error("Error getting current position:", error);
          // Default position if geolocation fails (Hanoi, Vietnam)
          setPosition([21.0285, 105.8542]);
        }
      );
    } else {
      // Default position if geolocation is not supported
      setPosition([21.0285, 105.8542]);
    }
  };

  // Set initial position based on user's location
  useEffect(() => {
    getCurrentPosition();
  }, []);

  // Update position when coordinates prop changes
  useEffect(() => {
    if (coordinates.latitude !== null && coordinates.longitude !== null) {
      // Normalize coordinates from props
      const [normalizedLat, normalizedLng] = normalizeCoordinates(
        coordinates.latitude,
        coordinates.longitude
      );
      setPosition([normalizedLat, normalizedLng]);
    }
  }, [coordinates.latitude, coordinates.longitude]);

  // Handle marker position change
  const handleMarkerPositionChange = (lat: number, lng: number) => {
    // Normalize coordinates
    const [normalizedLat, normalizedLng] = normalizeCoordinates(lat, lng);
    setPosition([normalizedLat, normalizedLng]);
    onCoordinateSelect(normalizedLat, normalizedLng);
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

  // Keep focus on search input when search results change
  useEffect(() => {
    if (searchResults.length > 0 && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [searchResults]);

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

        // Normalize coordinates
        const [normalizedLat, normalizedLng] = normalizeCoordinates(
          centroidLat,
          centroidLon
        );

        // Update marker position with centroid
        setPosition([normalizedLat, normalizedLng]);
        onCoordinateSelect(normalizedLat, normalizedLng);
      } else {
        // Fallback to regular coordinates
        const [normalizedLat, normalizedLng] = normalizeCoordinates(
          latitude,
          longitude
        );
        setPosition([normalizedLat, normalizedLng]);
        onCoordinateSelect(normalizedLat, normalizedLng);
      }
    } else {
      // If osm_type or osm_id not available, use regular coordinates
      const [normalizedLat, normalizedLng] = normalizeCoordinates(
        latitude,
        longitude
      );
      setPosition([normalizedLat, normalizedLng]);
      onCoordinateSelect(normalizedLat, normalizedLng);
    }

    // Update search query but keep the popover open
    setSearchQuery(result.display_name);

    // Keep focus on the input and keep popover open
    if (searchInputRef.current) {
      searchInputRef.current.focus();
    }
  };

  // Handle search form submission
  const handleSearchSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // If search query is empty, use current GPS position
    if (searchQuery.trim() === "") {
      if (currentGpsPosition) {
        setPosition(currentGpsPosition);
        onCoordinateSelect(currentGpsPosition[0], currentGpsPosition[1]);
        mapRef.current?.setView(currentGpsPosition, 14);
      } else {
        // If GPS position isn't available yet, try to get it
        getCurrentPosition();
      }
      return;
    }

    if (searchResults.length > 0) {
      const firstResult = searchResults[0];
      const latitude = Number.parseFloat(firstResult.lat);
      const longitude = Number.parseFloat(firstResult.lon);

      // If details are available, prioritize centroid
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

          // Normalize coordinates
          const [normalizedLat, normalizedLng] = normalizeCoordinates(
            centroidLat,
            centroidLon
          );

          setPosition([normalizedLat, normalizedLng]);
          onCoordinateSelect(normalizedLat, normalizedLng);
          mapRef.current?.setView([normalizedLat, normalizedLng], 14);
          setIsPopoverOpen(false);
          return;
        }
      }

      // If no centroid available, use original lat/lon
      const [normalizedLat, normalizedLng] = normalizeCoordinates(
        latitude,
        longitude
      );
      setPosition([normalizedLat, normalizedLng]);
      onCoordinateSelect(normalizedLat, normalizedLng);
      mapRef.current?.setView([normalizedLat, normalizedLng], 14);
      setSearchQuery(firstResult.display_name);
      setIsPopoverOpen(false);
    } else {
      alert("Không tìm thấy địa điểm phù hợp.");
      // Re-focus the input after alert
      if (searchInputRef.current) {
        searchInputRef.current.focus();
      }
    }
  };

  // Effect to maintain focus on search input except when user deliberately clicks away
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      // Keep focus on the search input unless user is clicking somewhere else deliberately
      if (
        searchInputRef.current &&
        e.target instanceof Node &&
        !searchInputRef.current.contains(e.target) &&
        !(e.target as Element).closest(".popover-content")
      ) {
        // User clicked outside the search input and popover, let them
      } else if (searchInputRef.current) {
        searchInputRef.current.focus();
      }
    };

    document.addEventListener("click", handleClick);

    return () => {
      document.removeEventListener("click", handleClick);
    };
  }, []);

  if (!position) {
    return (
      <div className="flex items-center justify-center h-[500px] bg-gray-100">
        <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
        <span className="ml-2 text-gray-500">Đang tải bản đồ...</span>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <form
        onSubmit={handleSearchSubmit}
        className="flex gap-2 mb-2 relative z-50"
      >
        <div className="relative flex-1">
          <Popover
            open={isPopoverOpen}
            onOpenChange={(open) => {
              setIsPopoverOpen(open);
              // When popover opens or closes, refocus the input
              if (searchInputRef.current && !open) {
                searchInputRef.current.focus();
              }
            }}
          >
            <PopoverTrigger asChild>
              <div className="w-full">
                <Input
                  type="text"
                  placeholder="Tìm kiếm địa điểm..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full"
                  ref={searchInputRef}
                  // Prevent the input from losing focus when the popover opens
                  onFocus={() => {
                    if (searchResults.length > 0) {
                      setIsPopoverOpen(true);
                    }
                  }}
                />
              </div>
            </PopoverTrigger>
            <PopoverContent
              className="p-0 w-[300px] lg:w-[400px] popover-content"
              align="start"
              style={styles.popoverContentStyle}
              sideOffset={5}
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
          // Add worldCopyJump to handle coordinates properly across date line
          worldCopyJump={true}
        >
          <LayersControl position="topright">
            <LayersControl.BaseLayer checked name={mapLayers.basic.name}>
              <TileLayer
                url={mapLayers.basic.url}
                attribution={mapLayers.basic.attribution}
              />
            </LayersControl.BaseLayer>

            <LayersControl.BaseLayer name={mapLayers.hybrid.name}>
              <TileLayer
                url={mapLayers.hybrid.url}
                attribution={mapLayers.hybrid.attribution}
              />
            </LayersControl.BaseLayer>
          </LayersControl>

          {position && (
            <Marker
              position={position}
              icon={customIcon}
              draggable={true}
              eventHandlers={{
                dragend: (e) => {
                  const marker = e.target;
                  const position = marker.getLatLng();
                  // Normalize coordinates from drag event
                  const [normalizedLat, normalizedLng] = normalizeCoordinates(
                    position.lat,
                    position.lng
                  );
                  handleMarkerPositionChange(
                    Number.parseFloat(normalizedLat.toFixed(6)),
                    Number.parseFloat(normalizedLng.toFixed(6))
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
