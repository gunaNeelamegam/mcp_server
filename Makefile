.PHONY: install ingest up down mcp-up mcp-down status test eval clean

PYTHON := uv run python
RUN_DIR := .run
PIDFILE := $(RUN_DIR)/mcp.pids

MCP_SERVERS := \
	mcp_servers.shipments.server:7001 \
	mcp_servers.inbound.server:7002 \
	mcp_servers.outbound.server:7003 \
	mcp_servers.inventory.server:7004 \
	mcp_servers.returns.server:7005

# --- Setup ---------------------------------------------------------------

install:
	uv sync

# --- Ingestion -------------------------------------------------------------
# Qdrant runs in embedded (on-disk) mode: only one process can hold
# ./qdrant_storage at a time. `ingest` must not run while `make up` is
# running — stop the app first (Ctrl+C / `make down`), then ingest.

ingest:
	$(PYTHON) -m ingestion.cli sync

status:
	$(PYTHON) -m ingestion.cli status

# --- MCP servers -----------------------------------------------------------

mcp-up: $(RUN_DIR)
	@if [ -f $(PIDFILE) ]; then echo "MCP servers already running (found $(PIDFILE)); run 'make mcp-down' first"; exit 1; fi
	@for entry in $(MCP_SERVERS); do \
		module=$${entry%:*}; port=$${entry#*:}; \
		echo "starting $$module on :$$port"; \
		uv run uvicorn $$module:app --port $$port --log-level warning \
			> $(RUN_DIR)/mcp-$$port.log 2>&1 & \
		echo $$! >> $(PIDFILE); \
	done
	@echo "waiting for MCP servers to become ready..."
	@for entry in $(MCP_SERVERS); do \
		port=$${entry#*:}; \
		for i in $$(seq 1 30); do \
			curl -sf http://localhost:$$port/tools > /dev/null 2>&1 && break; \
			sleep 0.5; \
		done; \
		curl -sf http://localhost:$$port/tools > /dev/null 2>&1 || { echo "port $$port did not come up, see $(RUN_DIR)/mcp-$$port.log"; exit 1; }; \
	done
	@echo "all 5 MCP servers ready."

mcp-down:
	@if [ -f $(PIDFILE) ]; then xargs kill < $(PIDFILE) 2>/dev/null || true; rm -f $(PIDFILE); fi
	@pkill -f "uvicorn mcp_servers\.(shipments|inbound|outbound|inventory|returns)\.server:app" 2>/dev/null || true
	@echo "MCP servers stopped."

$(RUN_DIR):
	mkdir -p $(RUN_DIR)

# --- App ---------------------------------------------------------------
# One-command start: brings up the 5 MCP servers in the background, then
# runs the main app in the foreground so you see live logs. Ctrl+C stops
# the app AND cleans up the background MCP servers (trap below).

up: mcp-up
	@trap '$(MAKE) mcp-down' EXIT INT TERM; \
	uv run uvicorn app.main:app --port 8010 --reload; \
	$(MAKE) mcp-down

down: mcp-down
	@echo "if the app server (port 8010) is still running, stop it with Ctrl+C in its terminal."

# --- Tests / eval -----------------------------------------------------------

test:
	uv run pytest -q

eval:
	$(PYTHON) -m tests.rag_eval.run_eval

# --- Cleanup -----------------------------------------------------------

clean: mcp-down
	rm -rf $(RUN_DIR)
