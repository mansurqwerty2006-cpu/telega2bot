# Telegram AI Bot

Простой Telegram-бот с ответами через Gemini, поиском через Tavily и погодой через OpenWeather.

## Запуск

1. Установи Python 3.10 или новее.
2. Открой PowerShell в папке проекта.
3. Запусти:

```powershell
.\run.ps1
```

Скрипт запустит бота. Внешние Python-зависимости не нужны.

## Настройки

Токены лежат в `.env`:

```env
TELEGRAM_BOT_TOKEN=...
GEMINI_API_KEY=...
TAVILY_API_KEY=...
OPENWEATHER_API_KEY=...
GEMINI_MODEL=gemini-3.5-flash
MAX_OUTPUT_TOKENS=2048
```

`.env` добавлен в `.gitignore`, чтобы случайно не опубликовать ключи.

## Команды бота

- `/start` - приветствие
- `/help` - помощь
- `/reset` - очистить контекст диалога
- `/search запрос` - поиск в интернете через Tavily
- `/weather город` - текущая погода через OpenWeather
- `/model` - показать активную модель

## Проверка

Перед запуском можно проверить конфигурацию:

```powershell
python bot.py --check
```

## Render

Проект можно задеплоить как Render Web Service. В Render нужны переменные:

- `TELEGRAM_BOT_TOKEN`
- `GEMINI_API_KEY`
- `TAVILY_API_KEY`
- `OPENWEATHER_API_KEY`
- `GEMINI_MODEL`
- `MAX_OUTPUT_TOKENS`

Бот слушает `PORT` только для Render health check и параллельно запускает Telegram polling.
Для постоянной работы лучше использовать Render Background Worker, но он может требовать карту.

## Важно

Бот работает, пока запущен `python bot.py` или `.\run.ps1`. Для постоянной работы его нужно держать на сервере/VPS или хостинге, который поддерживает Python-процессы.
