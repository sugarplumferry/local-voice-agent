param([string]$Command = "help")

$ErrorActionPreference = "Stop"

switch ($Command) {

    "up" {
        Write-Host "Starting all containers..." -ForegroundColor Cyan
        docker compose up -d
    }

    "down" {
        Write-Host "Stopping all containers..." -ForegroundColor Cyan
        docker compose down
    }

    "rebuild" {
        # Use this when requirements.txt or Dockerfile changes
        Write-Host "Rebuilding backend image and restarting..." -ForegroundColor Cyan
        docker compose up -d --build backend
    }

    "restart" {
        # Use this when only .env changes (no code change, no image rebuild needed)
        Write-Host "Restarting backend container..." -ForegroundColor Cyan
        docker compose restart backend
    }

    "frontend" {
        # Recreate frontend container — needed when nginx.conf or docker-compose volumes change
        Write-Host "Recreating frontend container..." -ForegroundColor Cyan
        docker compose up -d --force-recreate frontend
    }

    "logs" {
        Write-Host "Following backend logs (Ctrl+C to exit)..." -ForegroundColor Cyan
        docker compose logs -f backend
    }

    "test" {
        Write-Host "Running pytest inside backend container..." -ForegroundColor Cyan
        docker exec voice-agent-backend python -m pytest tests/ -v
    }

    "studio" {
        Write-Host "Starting LangGraph Studio inside backend container..." -ForegroundColor Cyan
        docker compose restart backend | Out-Null
        Start-Sleep -Seconds 5
        docker exec -d voice-agent-backend sh -c "cd /app && langgraph dev --host 0.0.0.0 --port 2024 > /tmp/studio.log 2>&1"
        Start-Sleep -Seconds 5
        Write-Host ""
        Write-Host "Open Studio:" -ForegroundColor Yellow
        Write-Host "  https://smith.langchain.com/studio/?baseUrl=http://localhost:2024" -ForegroundColor Green
        Write-Host ""
        Write-Host "Logs: docker exec voice-agent-backend cat /tmp/studio.log" -ForegroundColor Gray
    }

    "ngrok" {
        # Start ngrok tunnel on port 3000 in a new window, print the public URL
        if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
            Write-Host "ngrok not found. Install it: winget install ngrok" -ForegroundColor Red
            exit 1
        }
        Write-Host "Starting ngrok tunnel on port 3000..." -ForegroundColor Cyan
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "ngrok http 3000"
        Write-Host ""
        Write-Host "ngrok opened in a new window." -ForegroundColor Green
        Write-Host "Copy the https:// URL from that window and open it on your phone." -ForegroundColor Yellow
        Write-Host "Run '.\dev.ps1 ngrok-stop' to kill the tunnel." -ForegroundColor Gray
    }

    "ngrok-stop" {
        Write-Host "Stopping all ngrok processes..." -ForegroundColor Cyan
        $procs = Get-Process -Name ngrok -ErrorAction SilentlyContinue
        if ($procs) {
            $procs | Stop-Process -Force
            Write-Host "ngrok stopped." -ForegroundColor Green
        } else {
            Write-Host "No ngrok process found." -ForegroundColor Yellow
        }
    }

    "phone" {
        # Start containers + ngrok in one shot
        Write-Host "Starting containers..." -ForegroundColor Cyan
        docker compose up -d
        if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
            Write-Host "Containers up. ngrok not found — install with: winget install ngrok" -ForegroundColor Yellow
            exit 0
        }
        Write-Host "Starting ngrok tunnel on port 3000..." -ForegroundColor Cyan
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "ngrok http 3000"
        Write-Host ""
        Write-Host "Everything started." -ForegroundColor Green
        Write-Host "Copy the https:// URL from the ngrok window and open it on your phone." -ForegroundColor Yellow
    }

    "phone-stop" {
        # Stop ngrok then stop containers
        Write-Host "Stopping ngrok..." -ForegroundColor Cyan
        $procs = Get-Process -Name ngrok -ErrorAction SilentlyContinue
        if ($procs) { $procs | Stop-Process -Force; Write-Host "ngrok stopped." -ForegroundColor Green }
        Write-Host "Stopping containers..." -ForegroundColor Cyan
        docker compose down
    }

    "ps" {
        docker compose ps
    }

    "help" {
        Write-Host ""
        Write-Host "Usage: .\dev.ps1 <command>" -ForegroundColor White
        Write-Host ""
        Write-Host "Daily use:" -ForegroundColor Yellow
        Write-Host "  up           Start all containers"
        Write-Host "  down         Stop all containers"
        Write-Host "  ps           Show container status"
        Write-Host "  logs         Follow backend logs"
        Write-Host ""
        Write-Host "Phone access (ngrok):" -ForegroundColor Yellow
        Write-Host "  phone        Start containers + ngrok tunnel (all-in-one)"
        Write-Host "  phone-stop   Stop ngrok + containers"
        Write-Host "  ngrok        Start ngrok tunnel only (containers already running)"
        Write-Host "  ngrok-stop   Kill ngrok process"
        Write-Host ""
        Write-Host "Development:" -ForegroundColor Yellow
        Write-Host "  rebuild      Rebuild backend image  (use after requirements.txt changes)"
        Write-Host "  restart      Restart backend only   (use after .env changes)"
        Write-Host "  frontend     Recreate frontend      (use after nginx.conf changes)"
        Write-Host "  test         Run pytest inside the backend container"
        Write-Host "  studio       Start LangGraph Studio"
        Write-Host ""
        Write-Host "Hot-reload (no command needed):" -ForegroundColor Gray
        Write-Host "  backend/*.py  ->  uvicorn auto-reloads via WatchFiles"
        Write-Host "  frontend/*    ->  nginx serves from volume, refresh browser"
        Write-Host ""
    }

    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Run .\dev.ps1 help to see available commands."
    }
}
