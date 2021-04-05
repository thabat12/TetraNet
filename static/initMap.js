// [START maps_add_map]
// Initialize and add the map
function initMap() {
  // [START maps_add_map_instantiate_map]
  // The location of Uluru
  const uluru = {lat: 22.298337, lng: 114.17};
  // The map, centered at Uluru
  const map = new google.maps.Map(document.getElementById("map"), {
    zoom: 16,
    center: uluru,
  });
  // [END maps_add_map_instantiate_map]
  // [START maps_add_map_instantiate_marker]
  // The marker, positioned at Uluru
  const marker = new google.maps.Marker({
    position: uluru,
    map: map,
  });
  // [END maps_add_map_instantiate_marker]
}

// [END maps_add_map]
