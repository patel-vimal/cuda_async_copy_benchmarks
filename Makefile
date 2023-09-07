.PHONY: all build debug clean profile bench cuobjdump

CMAKE := cmake

BUILD_DIR := build
BENCHMARK_DIR := benchmark_results

all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Release ..
	@$(MAKE) -C $(BUILD_DIR)

debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug ..
	@$(MAKE) -C $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)

FUNCTION := $$(echo $$(cuobjdump -symbols build/main | grep -i sum_ | awk '{print $$NF}') | sed -e 's/\s\+/,/g')

cuobjdump: build
	@cuobjdump -arch sm_80 -sass -fun $(FUNCTION) build/main | c++filt > build/cuobjdump.sass
	@cuobjdump -arch sm_80 -ptx -fun $(FUNCTION) build/main | c++filt > build/cuobjdump.ptx

# Usage: make profile KERNEL=<integer> PREFIX=<optional string>
profile: build
	@mkdir -p $(BENCHMARK_DIR)
	@ncu --set full --export $(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL) --force-overwrite $(BUILD_DIR)/main $(KERNEL)

debug_profile: debug
	@mkdir -p $(BENCHMARK_DIR)
	@ncu --set full --export $(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL)_debug --force-overwrite $(BUILD_DIR)/main $(KERNEL)
