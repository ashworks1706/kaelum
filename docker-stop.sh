#!/bin/bash

echo "Stopping Kaelum services..."
docker-compose down

echo ""
echo "✓ Stopped"
echo ""
echo "To remove volumes (deletes cached data):"
echo "  docker-compose down -v"
